import cv2
import torch
import pytorch_lightning as pl


from lib.models.rscr.encoder.acebackbone import AceBackbone
from lib.models.rscr.head import *
from lib.models.regression.aggregator import *
from lib.models.regression.head import *
from lib.models.matching.pose_solver import PnPSolver

from lib.utils.loss import *
from lib.utils.metrics import pose_error_torch, error_auc, A_metrics


def project_3dto2d(xyz_B3N, K_B33, DEPTH_MIN):
    B, _, N = xyz_B3N.shape
    assert xyz_B3N.shape[1] == 3, f"Expect xyz_B3N.shape[1] == 3, got {xyz_B3N.shape[1]}"
    assert K_B33.shape == (B, 3, 3), f"Expect K_B33.shape == ({B}, 3, 3), got {K_B33.shape}"

    uv_B3N = torch.bmm(K_B33, xyz_B3N)
    # Avoid division by zero.
    # NOTE: negative values are also clamped at +DEPTH_MIN. The predicted pixel would be wrong,
    # but that's fine since we mask them out later.
    uv_B3N[:, 2].clamp_(min=DEPTH_MIN)
    # Dehomogenize
    uv_B2N = uv_B3N[:, :2] / uv_B3N[:, 2:3]
    return uv_B2N


class BaseRSCRModel(pl.LightningModule):
    """Regresses Relative Scene Coordinates between a pair of images"""

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        # initialise encoder
        try:
            encoder = eval(cfg.ENCODER.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid encoder {cfg.ENCODER.TYPE}')
        self.encoder = encoder(cfg.ENCODER)

        # aggregator
        try:
            aggregator = eval(cfg.AGGREGATOR.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid aggregator {cfg.AGGREGATOR.TYPE}')
        self.aggregator = aggregator(cfg.AGGREGATOR, self.encoder.num_out_layers)

        # head
        try:
            head = eval(cfg.HEAD.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid head {cfg.HEAD.TYPE}')
        self.head = head(cfg, self.aggregator.num_out_layers)

        # initialise pose loss function
        try:
            self.rot_loss = eval(cfg.TRAINING.ROT_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid rotation loss {cfg.TRAINING.ROT_LOSS}')
        try:
            self.trans_loss = eval(cfg.TRAINING.TRANS_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid translation loss {cfg.TRAINING.TRANS_LOSS}')
        # set loss function weights
        # if LAMBDA is 0., use learnable weights from
        # Geometric loss functions for camera pose regression with deep learning (Kendal & Cipolla)
        # self.LAMBDA = cfg.TRAINING.LAMBDA
        # if cfg.TRAINING.LAMBDA == 0.:
        #     self.s_r = torch.nn.Parameter(torch.zeros(1))
        #     self.s_t = torch.nn.Parameter(torch.zeros(1))

        # initialise pose solver
        # FIXME: do not allow to config this? since we only support PnP now
        if cfg.POSE_SOLVER == 'PNP':
            self.pose_solver = PnPSolver(cfg)
        else:
            raise NotImplementedError(f'Invalid pose solver {cfg.POSE_SOLVER}')

        # NOTE: delay to compute uv_grid until we have the first batch
        # FIXME: is there something like static attribute in python?
        self.single_uv_grid = None
    
    def training_step(self, batch, batch_idx):
        self.forward_pose(batch)
        loss = self.loss_fn(batch)
        R_loss = self.rot_loss(batch)
        t_loss = self.trans_loss(batch)

        self.log('train/R_loss', R_loss)
        self.log('train/t_loss', t_loss)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        Tgt = batch['T_0to1']
        R, t, n_inliers = self.forward_pose(batch)
        loss = self.loss_fn(batch)
        R_loss = self.rot_loss(batch)
        t_loss = self.trans_loss(batch)

        # validation metrics
        outputs = pose_error_torch(R, t, Tgt, reduce=None)
        outputs['R_loss'] = R_loss
        outputs['t_loss'] = t_loss
        outputs['loss'] = loss
        return outputs

    def validation_epoch_end(self, outputs):
        # aggregates metrics/losses from all validation steps
        aggregated = {}
        for key in outputs[0].keys():
            aggregated[key] = torch.stack([x[key] for x in outputs])

        # compute stats
        median_t_ang_err = aggregated['t_err_ang'].median()
        median_t_scale_err = aggregated['t_err_scale'].median()
        median_t_euclidean_err = aggregated['t_err_euc'].median()
        median_R_err = aggregated['R_err'].median()
        mean_R_loss = aggregated['R_loss'].mean()
        mean_t_loss = aggregated['t_loss'].mean()
        mean_loss = aggregated['loss'].mean()

        # a1, a2, a3 metrics of the translation vector norm
        a1, a2, a3 = A_metrics(aggregated['t_err_scale_sym'])

        # compute AUC of Euclidean translation error for 10cm, 50cm and 1m thresholds
        AUC_euc_10, AUC_euc_50, AUC_euc_100 = error_auc(
            aggregated['t_err_euc'].view(-1).detach().cpu().numpy(),
            [0.1, 0.5, 1.0]).values()

        # compute AUC of pose error (max of rot and t ang. error) for 5, 10 and 20 degrees thresholds
        pose_error = torch.maximum(
            aggregated['t_err_ang'].view(-1),
            aggregated['R_err'].view(-1)).detach().cpu()
        AUC_pos_5, AUC_pos_10, AUC_pos_20 = error_auc(pose_error.numpy(), [5, 10, 20]).values()

        # compute AUC of rotation error 5, 10 and 20 deg thresholds
        rot_error = aggregated['R_err'].view(-1).detach().cpu()
        AUC_rot_5, AUC_rot_10, AUC_rot_20 = error_auc(rot_error.numpy(), [5, 10, 20]).values()

        # compute AUC of translation angle error 5, 10 and 20 deg thresholds
        t_ang_error = aggregated['t_err_ang'].view(-1).detach().cpu()
        AUC_tang_5, AUC_tang_10, AUC_tang_20 = error_auc(t_ang_error.numpy(), [5, 10, 20]).values()

        # log stats
        self.log('val_loss/R_loss', mean_R_loss)
        self.log('val_loss/t_loss', mean_t_loss)
        self.log('val_loss/loss', mean_loss)
        self.log('val_metrics/t_ang_err', median_t_ang_err)
        self.log('val_metrics/t_scale_err', median_t_scale_err)
        self.log('val_metrics/t_euclidean_err', median_t_euclidean_err)
        self.log('val_metrics/R_err', median_R_err)
        self.log('val_auc/euc_10', AUC_euc_10)
        self.log('val_auc/euc_50', AUC_euc_50)
        self.log('val_auc/euc_100', AUC_euc_100)
        self.log('val_auc/pose_5', AUC_pos_5)
        self.log('val_auc/pose_10', AUC_pos_10)
        self.log('val_auc/pose_20', AUC_pos_20)
        self.log('val_auc/rot_5', AUC_rot_5)
        self.log('val_auc/rot_10', AUC_rot_10)
        self.log('val_auc/rot_20', AUC_rot_20)
        self.log('val_auc/tang_5', AUC_tang_5)
        self.log('val_auc/tang_10', AUC_tang_10)
        self.log('val_auc/tang_20', AUC_tang_20)
        self.log('val_t_scale/a1', a1)
        self.log('val_t_scale/a2', a2)
        self.log('val_t_scale/a3', a3)

        return mean_loss

    def configure_optimizers(self):
        tcfg = self.cfg.TRAINING
        opt = torch.optim.AdamW(self.parameters(), lr=tcfg.LR, eps=1e-6, amsgrad=True)
        if tcfg.LR_STEP_INTERVAL:
            scheduler = torch.optim.lr_scheduler.StepLR(
                opt, tcfg.LR_STEP_INTERVAL, tcfg.LR_STEP_GAMMA)
            return {'optimizer': opt, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        return opt
    
    def forward_pose(self, data):
        self(data)
        R, t, n_inliers = self.solve_pose_with_pnp(data)
        data['R'] = R
        data['t'] = t
        data['n_inliers'] = n_inliers
        return R, t, n_inliers

    def solve_pose_with_pnp(self, data):
        assert 'image0' in data, '"image0" is not in data'
        assert 'K_color1' in data, '"K_color1" is not in data'
        assert 'xyz1_0_B3HW' in data, '"xyz1_0_B3HW" is not in data'

        B = data['image0'].shape[0]

        self._set_single_uv_grid(data)
        uv_grid = self.single_uv_grid.expand(B, -1, -1, -1)

        Rs = torch.empty(B, 3, 3).to(data['image0'].device)
        ts = torch.empty(B, 1, 3).to(data['image0'].device)
        n_inliers = torch.empty(B, 1).to(data['image0'].device)

        with torch.no_grad():
            batch_xyz_0 = data['xyz1_0_B3HW'].view(B, 3, -1).transpose(1, 2) # (B, N, 3)
            batch_uv_1 = uv_grid.view(B, 2, -1).transpose(1, 2) # (B, N, 2)
            batch_K1 = data['K_color1'].view(B, 3, 3)
            for idx in range(B):
                xyz_0 = batch_xyz_0[idx].cpu().numpy()
                uv_1 = batch_uv_1[idx].cpu().numpy()
                K1 = batch_K1[idx].cpu().numpy()

                R, t, n_inlier = self.pose_solver.estimate_pose(xyz_0, uv_1, K1)
                Rs[idx] = torch.from_numpy(R)
                ts[idx] = torch.from_numpy(t).view(1, 3)
                n_inliers[idx] = n_inlier

        return Rs, ts, n_inliers
    
    def _set_single_uv_grid(self, data):
        assert 'image0' in data, '"image0" is not in data'
        assert 'xyz1_0_B3HW' in data, '"xyz1_0_B3HW" is not in data'

        B, imD, imH, imW = data['image0'].shape
        _, outD, outH, outW = data['xyz1_0_B3HW'].shape

        is_updated = False
        if self.single_uv_grid is None or self.single_uv_grid.shape != (1, 2, outH, outW):
            xs = torch.linspace(0, imW - 1, outW)
            ys = torch.linspace(0, imH - 1, outH)
            # NOTE: the 1st dim is for x (width), the 2nd dim is for y (height)
            grid = torch.meshgrid(ys, xs)
            uv_grid = torch.stack((grid[1], grid[0]), dim=0).float()
            uv_grid = uv_grid.unsqueeze(0).to(data['image0'].device)
            self.single_uv_grid = uv_grid
            is_updated = True

        assert self.single_uv_grid.shape == (1, 2, outH, outW)
        return is_updated
    
    def get_transformation_dict(self, data, 
                            return_uvgt_B2HW=False, 
                            return_xyz1_1_B3HW=False, 
                            return_uv1_1_B2HW=False):
        B, _, H, W = data['xyz1_0_B3HW'].shape
        xyz1_0_B3HW = data['xyz1_0_B3HW']
        xyz1_0_B3N = xyz1_0_B3HW.view(B, 3, -1)
        K1_B33 = data['K_color1'].float()
        Rgt_0to1 = data['T_0to1'][:, :3, :3]
        tgt_0to1 = data['T_0to1'][:, :3, 3:]
        Rgt_1to0 = Rgt_0to1.transpose(1, 2)
        tgt_1to0 = -torch.bmm(Rgt_1to0, tgt_0to1)
        assert K1_B33.shape == (B, 3, 3)
        assert Rgt_0to1.shape == (B, 3, 3)
        assert tgt_0to1.shape == (B, 3, 1)
        assert Rgt_1to0.shape == (B, 3, 3)
        assert tgt_1to0.shape == (B, 3, 1)

        if return_uvgt_B2HW:
            self._set_single_uv_grid(data)
            uvgt_B2HW = self.single_uv_grid.expand(B, -1, -1, -1)

        if return_xyz1_1_B3HW or return_uv1_1_B2HW:
            xyz1_1_B3N = torch.bmm(Rgt_0to1, xyz1_0_B3N) + tgt_0to1
            xyz1_1_B3HW = xyz1_1_B3N.view(B, 3, H, W)
        
        if return_uv1_1_B2HW:
            uv1_1_B2N = project_3dto2d(xyz1_1_B3N, K1_B33, DEPTH_MIN=self.cfg.TRAINING.CUSTOM_LOSS.DEPTH_MIN)
            uv1_1_B2HW = uv1_1_B2N.view(B, 2, H, W)

        ret = { 'xyz1_0_B3HW': xyz1_0_B3HW, }
        if return_uvgt_B2HW:
            ret['uvgt_B2HW'] = uvgt_B2HW
        if return_xyz1_1_B3HW:
            ret['xyz1_1_B3HW'] = xyz1_1_B3HW
        if return_uv1_1_B2HW:
            ret['uv1_1_B2HW'] = uv1_1_B2HW

        return ret


class RSCRModel(BaseRSCRModel):
    def __init__(self, cfg): 
        super().__init__(cfg)

        if cfg.ENCODER.SHOULD_LOAD_PRETRAINED:
            assert cfg.ENCODER.PRETRAINED_PATH is not None, 'PRETRAINED_PATH must be set when SHOULD_LOAD_PRETRAINED is True'
            assert cfg.ENCODER.SHOULD_FREEZE_PRETRAINED is not None, 'SHOULD_FREEZE_PRETRAINED must be set when SHOULD_LOAD_PRETRAINED is True'
            try:
                self.encoder.load_state_dict(torch.load(cfg.ENCODER.PRETRAINED_PATH))
            except:
                raise RuntimeError(f'Failed to load pretrained weights from {cfg.ENCODER.PRETRAINED_PATH}')
            if cfg.ENCODER.SHOULD_FREEZE_PRETRAINED:
                self.encoder.freeze()

        # initialise custom loss function
        try:
            self.custom_loss = eval(cfg.TRAINING.CUSTOM_LOSS.TYPE)
        except NameError:
            raise NotImplementedError(f'Invalid custom loss {cfg.TRAINING.CUSTOM_LOSS.TYPE}')

    def forward(self, data):
        vol0 = self.encoder(data['image0'])
        vol1 = self.encoder(data['image1'])
        volume_q1k0 = self.aggregator(vol1, vol0)
        # xyz1_0_B3HW = self.head(volume_q1k0)
        xyz1_0_B3HW = self.head(torch.cat((volume_q1k0, vol1), dim=1))

        data['xyz1_0_B3HW'] = xyz1_0_B3HW
        return xyz1_0_B3HW

    def loss_fn(self, data):
        transformation_dict = self.get_transformation_dict(
            data, 
            return_uvgt_B2HW=True, 
            return_xyz1_1_B3HW=True, 
            return_uv1_1_B2HW=True)

        data['uvgt_B2HW'] = transformation_dict['uvgt_B2HW']
        data['xyz1_1_B3HW'] = transformation_dict['xyz1_1_B3HW']
        data['uv1_1_B2HW'] = transformation_dict['uv1_1_B2HW']

        data['loss_cfg'] = self.cfg.TRAINING.CUSTOM_LOSS
        data['current_optim_step'] = self.trainer.global_step
        data['total_optim_step'] = self.trainer.estimated_stepping_batches
        loss, info = self.custom_loss(data)

        # TODO: log this here seems weird?
        B = data['image0'].shape[0]
        valid_ratio = info['valid_count'] / info['total_count']
        self.log('train/valid_ratio', valid_ratio, batch_size=B)

        return loss
    