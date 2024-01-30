import numpy as np
import inspect
import logging

import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from kornia.geometry.conversions import rotation_matrix_to_quaternion, QuaternionCoeffOrder


def data_wrapper(func):
    """Decorator that obtains the functions arguments from the shared 'data' dictionary
    Allows loss functions to be self-contained without direct references to 'data' 
    """
    def wrapped(data):
        # get arguments names
        arg_list = list(inspect.signature(func).parameters)

        # fill dict with arguments names and their values (from shared data dict)
        arguments = {'R': data['R'],
                     't': data['t'],
                     'Rgt': data['T_0to1'][:, :3, :3],
                     'tgt': data['T_0to1'][:, :3, 3:].transpose(1, 2),
                     'K0': data['K_color0'].float(),
                     'K1': data['K_color1'].float(),
                    }

        # add quaternion ground-truth, if using quat. loss functions
        if 'q' in arg_list:
            arguments['q'] = data['q']
            qgt = rotation_matrix_to_quaternion(
                arguments['Rgt'].contiguous(),
                order=QuaternionCoeffOrder.WXYZ)
            # enforces using a single quaternion hemishehere (avoiding q, -q duble representation)
            qgt *= torch.sign(qgt[:, 0:1])
            arguments['qgt'] = qgt

        # add scale, if using specific loss functions
        if 'scale' in arg_list:
            arguments['scale'] = data['scale']
            arguments['scalegt'] = torch.linalg.norm(arguments['tgt'], dim=-1).unsqueeze(-1)

        # add t_direction, if using specific loss functions
        if 't_direction' in arg_list:
            arguments['t_direction'] = data['t_direction']
            arguments['t_directiongt'] = F.normalize(arguments['tgt'], dim=-1)

        # R_bins from AngularBin head
        if 'R_bins' in arg_list:
            arguments['R_bins'] = data['R_bins']
            R_binsgt = torch.from_numpy(
                Rotation.from_matrix(arguments['Rgt'].cpu().numpy()).as_euler(
                    'xyz', degrees=True))  # [B, 3]
            # add offset to get interval [0, 360] in XZ and [0,180] in Y
            R_binsgt += torch.FloatTensor([[180, 90, 180]])
            R_binsgt = torch.round(R_binsgt).long()
            R_binsgt[:, 0] = torch.clamp(R_binsgt[:, 0], 0, 359)  # clamps to fit in bins
            R_binsgt[:, 1] = torch.clamp(R_binsgt[:, 1], 0, 179)
            R_binsgt[:, 2] = torch.clamp(R_binsgt[:, 2], 0, 359)
            arguments['R_binsgt'] = R_binsgt.to(arguments['Rgt'].device)

        # spherical angles of translation vector, from AngularBin head
        if 't_sph_phi' in arg_list or 't_sph_theta' in arg_list:
            arguments['t_sph_phi'] = data['t_sph_phi']
            arguments['t_sph_theta'] = data['t_sph_theta']

            t_direction_gt = F.normalize(arguments['tgt'], dim=-1).reshape(-1, 3)
            t_sph_theta_gt = torch.acos(t_direction_gt[:, 2])
            t_sph_phi_gt = torch.atan2(t_direction_gt[:, 1], t_direction_gt[:, 0] + 1e-5)
            t_sph_phi_gt[t_sph_phi_gt < 0] += 2 * np.pi
            t_sph_theta_gt = torch.clamp(torch.round(torch.rad2deg(t_sph_theta_gt)).long(), 0, 179)
            t_sph_phi_gt = torch.round(torch.rad2deg(t_sph_phi_gt)).long()
            t_sph_phi_gt[t_sph_phi_gt == 360] = 0
            arguments['t_sph_phigt'] = t_sph_phi_gt
            arguments['t_sph_thetagt'] = t_sph_theta_gt
        
        if 'uvgt_B2HW' in arg_list:
            arguments['uvgt_B2HW'] = data['uvgt_B2HW']
        if 'xyz1_0_B3HW' in arg_list:
            arguments['xyz1_0_B3HW'] = data['xyz1_0_B3HW']
        if 'xyz1_1_B3HW' in arg_list:
            arguments['xyz1_1_B3HW'] = data['xyz1_1_B3HW']
        if 'uv1_1_B2HW' in arg_list:
            arguments['uv1_1_B2HW'] = data['uv1_1_B2HW']

        if 'lcfg' in arg_list:
            arguments['lcfg'] = data['loss_cfg']
        
        if 'current_optim_step' in arg_list:
            arguments['current_optim_step'] = data['current_optim_step']
        if 'total_optim_step' in arg_list:
            arguments['total_optim_step'] = data['total_optim_step']

        # get argument values and returns function result on arguments
        arg_value = [arguments[x] for x in arg_list]
        return func(*arg_value)

    return wrapped


@data_wrapper
def rot_frobenius_loss(R, Rgt):
    """Computes rotation loss using Frobenius norm.
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    B = R.shape[0]
    eye_batch = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(R.device)
    R_residual = Rgt.transpose(1, 2) @ R
    R_loss = F.mse_loss(R_residual, eye_batch)
    return R_loss


@data_wrapper
def rot_l1_loss(R, Rgt):
    """Computes rotation loss using L1 norm over residual rotation matrix.
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    B = R.shape[0]
    eye_batch = torch.eye(3).unsqueeze(0).repeat(B, 1, 1).to(R.device)
    R_residual = Rgt.transpose(1, 2) @ R
    R_loss = F.l1_loss(R_residual, eye_batch)
    return R_loss


@data_wrapper
def rot_angle_loss(R, Rgt):
    """
    Computes rotation loss using L2 error of residual rotation angle [radians]
    Input:
    R - estimated rotation matrix [B, 3, 3]
    Rgt - groundtruth rotation matrix [B, 3, 3]
    Output:  rotation_loss
    """

    residual = R.transpose(1, 2) @ Rgt
    trace = torch.diagonal(residual, dim1=-2, dim2=-1).sum(-1)
    cosine = (trace - 1) / 2
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    R_err = torch.acos(cosine)
    loss = F.l1_loss(R_err, torch.zeros_like(R_err))
    return loss


@data_wrapper
def rot_bin_loss(R_bins, R_binsgt):
    lrx = F.cross_entropy(R_bins[:, :360], R_binsgt[:, 0])
    lry = F.cross_entropy(R_bins[:, 360:540], R_binsgt[:, 1])
    lrz = F.cross_entropy(R_bins[:, 540:], R_binsgt[:, 2])
    return (lrx + lry + lrz) / 3


@data_wrapper
def trans_l2_loss(t, tgt):
    """Computes L2 loss for translation vector
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    return F.mse_loss(t, tgt)


@data_wrapper
def trans_l1_loss(t, tgt):
    """Computes L1 loss for translation vector
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    return F.l1_loss(t, tgt)


@data_wrapper
def quat_l1_loss(q, qgt):
    """Computes L1 loss between quaternions
    Input:
    q - estimated quaternion [B, 4]
    qgt - ground-truth quaternion [B, 4]
    Output: quat. loss
    """
    return F.l1_loss(q, qgt)


@data_wrapper
def robust_quat_l1_loss(q, qgt):
    """Robust L1 quaternion loss.

    q - estimated quaternion [B, 4]
    qgt - ground-truth quaternion [B, 4]

    Source: https://users.cecs.anu.edu.au/~hartley/Papers/PDF/Hartley-Trumpf:Rotation-averaging:IJCV.pdf
    page 10: "Quaternion distance"

    Note: probably assumes normalized quaternion, which for us is true for targ.

    Note2: min(||pred - targ||_2, ||pred + targ||_2)^2 would be a *non-robust* L2 loss.
    """
    assert q.shape[1] == 4
    assert qgt.shape[1] == 4
    return torch.mean(
        torch.minimum(torch.linalg.norm(q + qgt, dim=1, keepdim=True),
                      torch.linalg.norm(q - qgt, dim=1, keepdim=True)))


@data_wrapper
def trans_scale_direction_loss(scale, scalegt, t_direction, t_directiongt):
    """ Computes translation loss in two componentes: scale loss (L1) and t_direction loss (angular loss)
    Input:
    scale - estimated scale [B, 1, 1]
    scalegt - ground-truth scale [B, 1, 1]
    t_direction - estimated translation direction (unitary) [B, 1, 3]
    t_directiongt - ground-truth translation direction (unitary) [B, 1, 3]
    """
    return F.l1_loss(scale, scalegt) + F.l1_loss(t_direction, t_directiongt)


@data_wrapper
def trans_ang_loss(t, tgt):
    """Computes L1 loss for translation vector ANGULAR error
    Input:
    t - estimated translation vector [B, 1, 3]
    tgt - ground-truth translation vector [B, 1, 3]
    Output: translation_loss
    """

    scale_t = torch.linalg.norm(t, dim=-1)
    scale_tgt = torch.linalg.norm(tgt, dim=-1)

    cosine = (t @ tgt.transpose(1, 2)).squeeze(-1) / (scale_t * scale_tgt + 1e-6)
    cosine = torch.clip(cosine, -0.99999, 0.99999)  # handle numerical errors and NaNs
    t_ang_err = torch.acos(cosine)
    t_ang_err = torch.minimum(t_ang_err, np.pi - t_ang_err)
    return F.l1_loss(t_ang_err, torch.zeros_like(t_ang_err))


@data_wrapper
def trans_sphbin_loss(t_sph_phi, t_sph_phigt, t_sph_theta, t_sph_thetagt, scale, scalegt):
    lscale = F.l1_loss(scale, scalegt)
    lphi = F.cross_entropy(t_sph_phi, t_sph_phigt)
    ltheta = F.cross_entropy(t_sph_theta, t_sph_thetagt)
    return lscale + (lphi + ltheta) / 2


@data_wrapper
def trans_scale_l1_loss(scale, scalegt):
    return F.l1_loss(scale, scalegt)


@data_wrapper
def empty_loss(tgt):
    return torch.zeros(1, device=tgt.device, dtype=torch.float32)


@data_wrapper
def self_repro_loss(Rgt, tgt, K1, 
                    uvgt_B2HW, 
                    xyz1_0_B3HW, xyz1_1_B3HW, uv1_1_B2HW,
                    current_optim_step, total_optim_step, lcfg):
    """Computes self-reprojection loss between uv and uvgt
    """
    B, _, H, W = xyz1_0_B3HW.shape
    N = H * W

    K1_B33 = K1
    inv_K1_B33 = torch.inverse(K1_B33)
    R_0to1 = Rgt
    t_0to1 = tgt.transpose(1, 2)
    R_1to0 = R_0to1.transpose(1, 2)
    t_1to0 = -torch.bmm(R_1to0, t_0to1)
    assert K1_B33.shape == (B, 3, 3)
    assert R_0to1.shape == (B, 3, 3)
    assert t_0to1.shape == (B, 3, 1)
    assert R_1to0.shape == (B, 3, 3)
    assert t_1to0.shape == (B, 3, 1)
    assert uvgt_B2HW.shape == (B, 2, H, W), f'uvgt_B2HW.shape != (B, 2, H, W), got {uvgt_B2HW.shape}'

    uvgt_B2N = uvgt_B2HW.view(B, 2, -1)
    # self-reprojection
    xyz1_1_B3N = xyz1_1_B3HW.view(B, 3, -1)
    uv1_1_B2N = uv1_1_B2HW.view(B, 2, -1)

    # reprojection error
    repro_errs_BN = torch.norm(uv1_1_B2N - uvgt_B2N, dim=1, p=1)
    assert repro_errs_BN.shape == (B, N), f'repro_errs_BN.shape != ({B}, {N}), got {repro_errs_BN.shape}'

    # proxy 3D coordinate targets with constant depth assumption.
    dummy_xyz_B3N = torch.cat([uvgt_B2N, torch.ones_like(uvgt_B2N[:, :1])], dim=1)
    dummy_xyz_B3N = lcfg.DEPTH_TARGET * torch.bmm(inv_K1_B33, dummy_xyz_B3N)
    assert dummy_xyz_B3N.shape == (B, 3, N), f'dummy_xyz_B3N.shape != ({B}, 3, {N}), got {dummy_xyz_B3N.shape}'

    # === Compute masks for invalid pixels ===
    # Predicted coordinates behind or close to camera plane.
    # NOTE: negative depth is clamped at +DEPTH_MIN, so we need to mask them out.
    invalid_min_depth_BN = xyz1_1_B3N[:, 2] <= lcfg.DEPTH_MIN
    # Predicted coordinates beyond max distance.
    invalid_max_depth_BN = xyz1_1_B3N[:, 2] > lcfg.DEPTH_MAX
    # Very large reprojection errors.
    invalid_repro_BN = repro_errs_BN > lcfg.REPRO_HARD_CLAMP
    # Invalid mask is the union of all these. Valid mask is the opposite.
    invalid_mask_BN = (invalid_min_depth_BN | invalid_repro_BN | invalid_max_depth_BN)
    valid_mask_BN = ~invalid_mask_BN
    assert invalid_mask_BN.shape == (B, N)

    # Valid pixels: robust reprojection error
    valid_repro_errs = repro_errs_BN[valid_mask_BN]
    if lcfg.REPRO_TYPE == 'l1+sqrt':
        soft_clamp_mask = valid_repro_errs <= lcfg.REPRO_SOFT_CLAMP
        loss_valid_l1 = valid_repro_errs[soft_clamp_mask]
        loss_valid_sqrt = torch.sqrt(lcfg.REPRO_SOFT_CLAMP * valid_repro_errs[~soft_clamp_mask])
        valid_loss_cnt = len(loss_valid_l1) + len(loss_valid_sqrt)
        loss_valid  = loss_valid_l1.sum() + loss_valid_sqrt.sum()
    elif lcfg.REPRO_TYPE == 'tanh':
        valid_repro_errs = weighted_tanh(valid_repro_errs, lcfg.REPRO_SOFT_CLAMP)
        valid_loss_cnt = len(valid_repro_errs)
        loss_valid = valid_repro_errs.sum()
    elif lcfg.REPRO_TYPE == 'dyntanh':
        # FIXME: schedule based on epoch, not optim step, may not a good method
        schedule_weight = current_optim_step / total_optim_step
        # TODO: Optionally scale it if using the circular schedule
        schedule_weight = 1 - np.sqrt(1 - schedule_weight ** 2)
        weight = (1 - schedule_weight) * lcfg.REPRO_SOFT_CLAMP + lcfg.REPRO_SOFT_CLAMP_MIN
        valid_repro_errs = weighted_tanh(valid_repro_errs, weight)
        valid_loss_cnt = len(valid_repro_errs)
        loss_valid = valid_repro_errs.sum()
    elif lcfg.REPRO_TYPE == 'sc_init':
        # NOTE: use the proxy target for scene coordinate initialization
        invalid_mask_BN = torch.ones_like(invalid_mask_BN)
        valid_loss_cnt = 0
        loss_valid = torch.tensor([0])
    else:
        raise NotImplementedError(f'Unknown REPRO_TYPE: {lcfg.REPRO_TYPE}')

    # Invalid pixels: distance to the proxy 3D target
    loss_invalid = torch.abs(xyz1_1_B3N - dummy_xyz_B3N).sum(dim=1).masked_select(invalid_mask_BN)
    invalid_loss_cnt = len(loss_invalid)

    assert valid_loss_cnt + invalid_loss_cnt == B * N
    loss = loss_valid.sum() + loss_invalid.sum()
    loss = loss / (B * H * W)

    n_fails_in_loss_valid = torch.isnan(loss_valid).sum() + torch.isinf(loss_valid).sum()
    n_fails_in_loss_invalid = torch.isnan(loss_invalid).sum() + torch.isinf(loss_invalid).sum()
    if n_fails_in_loss_valid > 0:
        logging.warning(f'{n_fails_in_loss_valid} NaNs or INFs in loss_valid')
    if n_fails_in_loss_invalid > 0:
        logging.warning(f'{n_fails_in_loss_invalid} NaNs or INFs in loss_invalid')

    info = {
        'total_count': B * H * W,
        'valid_count': valid_loss_cnt,
        'invalid_count': invalid_loss_cnt,
    }

    return loss, info

def weighted_tanh(repro_errs, weight):
    return weight * torch.tanh(repro_errs / weight)
