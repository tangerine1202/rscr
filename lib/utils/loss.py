import math
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
                     'tgt': data['T_0to1'][:, :3, 3:].transpose(1, 2)
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
            t_sph_phi_gt[t_sph_phi_gt < 0] += 2 * math.pi
            t_sph_theta_gt = torch.clamp(torch.round(torch.rad2deg(t_sph_theta_gt)).long(), 0, 179)
            t_sph_phi_gt = torch.round(torch.rad2deg(t_sph_phi_gt)).long()
            t_sph_phi_gt[t_sph_phi_gt == 360] = 0
            arguments['t_sph_phigt'] = t_sph_phi_gt
            arguments['t_sph_thetagt'] = t_sph_theta_gt

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
    t_ang_err = torch.minimum(t_ang_err, math.pi - t_ang_err)
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

# RSCR loss


def self_repro_loss(data, uvgt_B2HW):
    """Computes self-reprojection loss between uv and uvgt
    Input:
    data - dictionary with keys:
        'cross_xyz' - RSC [B, 3, H, W]
        'self_uv' - reprojected uv coordinates of the same image [B, 2, H, W]
        'K_color1' - camera intrinsics [B, 3, 3]
    uvgt - ground-truth uv coordinates [B, 2, ...]
    Output: reprojection_loss
    """

    DEPTH_MIN = 0.1
    DEPTH_MAX = 1000
    DEPTH_TARGET = 10
    REPRO_LOSS_HARD_CLAMP = 1000
    # REPRO_LOSS_SOFT_CLAMP = 50

    B, _, H, W = uvgt_B2HW.shape
    N = H * W

    assert data['xyz1_0_B3HW'].shape == (B, 3, H, W)
    assert data['uv1_1_B2HW'].shape == (B, 2, H, W)
    assert data['K_color1'].shape == (B, 3, 3)

    cross_xyz_B3HW = data['xyz1_0_B3HW']
    cross_xyz_B3N = cross_xyz_B3HW.view(B, 3, -1)
    self_uv_B2HW = data['uv1_1_B2HW']
    self_uv_B2N = self_uv_B2HW.view(B, 2, -1)

    self_K_B33 = data['K_color1'].float()
    self_invK_B33 = torch.inverse(self_K_B33)

    uvgt_B2N = uvgt_B2HW.view(B, 2, -1)
    # Handle the invalid predictions: generate proxy coordinate targets with constant depth assumption.
    dummy_xyz_B3HW = torch.cat([uvgt_B2HW, torch.ones_like(uvgt_B2HW[:, :1])], dim=1)
    dummy_xyz_B3N = dummy_xyz_B3HW.view(B, 3, -1)
    dummy_xyz_B3N = DEPTH_TARGET * torch.bmm(self_invK_B33, dummy_xyz_B3N)

    assert self_uv_B2N.shape == (B, 2, N)
    assert cross_xyz_B3N.shape == (B, 3, N)
    assert self_K_B33.shape == (B, 3, 3)
    assert self_invK_B33.shape == (B, 3, 3)

    assert uvgt_B2HW.shape == (B, 2, H, W)
    assert dummy_xyz_B3HW.shape == (B, 3, H, W)
    assert dummy_xyz_B3N.shape == (B, 3, N)

    repro_errs_BN = torch.norm(self_uv_B2N - uvgt_B2N, dim=1, p=1)
    assert repro_errs_BN.shape == (B, N)

    #
    # Compute masks used to ignore invalid pixels.
    #
    # Predicted coordinates behind or close to camera plane.
    invalid_min_depth_BN = cross_xyz_B3N[:, 2] <= DEPTH_MIN
    # Predicted coordinates beyond max distance.
    invalid_max_depth_BN = cross_xyz_B3N[:, 2] > DEPTH_MAX
    # Very large reprojection errors.
    invalid_repro_BN = repro_errs_BN > REPRO_LOSS_HARD_CLAMP
    # Invalid mask is the union of all these. Valid mask is the opposite.
    invalid_mask_BN = (invalid_min_depth_BN | invalid_repro_BN | invalid_max_depth_BN)
    valid_mask_BN = ~invalid_mask_BN
    assert invalid_mask_BN.shape == (B, N)

    # valid reprojection error
    valid_repro_errs = repro_errs_BN[valid_mask_BN]
    loss_valid = valid_repro_errs

    # Compute the distance to target camera coordinates.
    loss_invalid = torch.norm(cross_xyz_B3N - dummy_xyz_B3N, dim=1, p=2).masked_select(invalid_mask_BN)

    assert len(loss_valid) + len(loss_invalid) == B * N
    n_fails_in_loss_valid = torch.isnan(loss_valid).sum() + torch.isinf(loss_valid).sum()
    n_fails_in_loss_invalid = torch.isnan(loss_invalid).sum() + torch.isinf(loss_invalid).sum()
    if n_fails_in_loss_valid > 0:
        logging.warning(f'{n_fails_in_loss_valid} NaNs or INFs in loss_valid')
    if n_fails_in_loss_invalid > 0:
        logging.warning(f'{n_fails_in_loss_invalid} NaNs or INFs in loss_invalid')

    loss = loss_valid.sum() + loss_invalid.sum()
    loss = loss / (B * N)

    info = {
        'total_count': B * N,
        'valid_count': len(loss_valid),
        'invalid_count': len(loss_invalid),
        'valid_mean': loss_valid.mean().item() if len(loss_valid) > 0 else 0.0,
        'invalid_mean': loss_invalid.mean().item() if len(loss_invalid) > 0 else 0.0,
    }

    return loss, info
