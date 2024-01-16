import cv2
import torch
import pytorch_lightning as pl


from lib.models.regression.model import RegressionModel 
from lib.models.regression.aggregator import *
from lib.models.regression.head import *
from lib.models.regression.encoder.resnet import ResNet
from lib.models.regression.encoder.resunet import ResUNet
from lib.models.regression.encoder.acebackbone import AceBackbone
from lib.models.matching.pose_solver import PnPSolver

from lib.utils.loss import *


class RSCRegressionModel(RegressionModel):
    """Regresses Relative Scene Coordinates between a pair of images"""

    def __init__(self, cfg):
        super().__init__(cfg)

        # load pretrained weights of encoder
        if cfg.ENCODER.SHOULD_LOAD_PRETRAINED:
            assert cfg.ENCODER.PRETRAINED_PATH is not None, 'PRETRAINED_PATH must be set when SHOULD_LOAD_PRETRAINED is True'
            assert cfg.ENCODER.SHOULD_FREEZE_PRETRAINED is not None, 'SHOULD_FREEZE_PRETRAINED must be set when SHOULD_LOAD_PRETRAINED is True'
            try:
                self.encoder.load_state_dict(torch.load(cfg.ENCODER.PRETRAINED_PATH))
            except:
                raise RuntimeError(f'Failed to load pretrained weights from {cfg.ENCODER.PRETRAINED_PATH}')
            if cfg.ENCODER.SHOULD_FREEZE_PRETRAINED:
                for param in self.encoder.parameters():
                    param.requires_grad = False

        try:
            self.self_repro_loss = eval(cfg.TRAINING.SELF_REPRO_LOSS)
        except NameError:
            raise NotImplementedError(f'Invalid self-reprojection loss {cfg.TRAINING.SELF_REPRO_LOSS}')

        if cfg.POSE_SOLVER == 'PNP':
            self.pose_solver = PnPSolver(cfg)
        else:
            raise NotImplementedError(f'Invalid pose solver {cfg.POSE_SOLVER}')

        # NOTE: delay to compute uv_grid until we have the first batch
        self.single_uv_grid = None

    def forward(self, data):
        vol0 = self.encoder(data['image0'])
        vol1 = self.encoder(data['image1'])
        volume_q1k0 = self.aggregator(vol1, vol0)
        xyz1_0_B3HW = self.head(volume_q1k0)

        data['out_HW'] = xyz1_0_B3HW.shape[-2:]
        data['xyz1_0_B3HW'] = xyz1_0_B3HW

        R, t = self._solve_pose(data)
        data['R'] = R
        data['t'] = t
        return R, t
    
    def loss_fn(self, data):
        xyz1_0_B3HW = data['xyz1_0_B3HW']
        B, _, outH, outW = xyz1_0_B3HW.shape
        K_0 = data['K_color0'].float().detach()
        K_1 = data['K_color1'].float().detach()
        T_0to1 = data['T_0to1'].float().detach()
        R_0to1 = T_0to1[:, :3, :3]
        t_0to1 = T_0to1[:, :3, 3:]
        assert K_0.shape == K_1.shape == (B, 3, 3)
        assert T_0to1.shape == (B, 4, 4)
        assert R_0to1.shape == (B, 3, 3)
        assert t_0to1.shape == (B, 3, 1)
        xyz1_0_B3N = xyz1_0_B3HW.view(B, 3, -1)

        # self-reprojection
        xyz1_1_B3N = torch.bmm(R_0to1, xyz1_0_B3N) + t_0to1
        uv1_1_B2N = self.project_3dto2d(K_1, xyz1_1_B3N)
        uv1_1_B2HW = uv1_1_B2N.view(B, 2, outH, outW)

        data['xyz1_0_B3HW'] = xyz1_0_B3HW
        data['uv1_1_B2HW'] = uv1_1_B2HW

        # FIXME: integrated into loss @data_wrapper
        self._set_single_uv_grid(data)
        uv_grid = self.single_uv_grid.expand(B, -1, -1, -1)

        self_loss, info = self.self_repro_loss(data, uv_grid)

        valid_ratio = info['valid_count'] / info['total_count']
        invalid_ratio = info['invalid_count'] / info['total_count']
        self.log('train/valid_ratio', valid_ratio, batch_size=B)
        self.log('train/invalid_ratio', invalid_ratio, batch_size=B)
        self.log('train/valid_mean', info['valid_mean'], batch_size=B)
        self.log('train/invalid_mean', info['invalid_mean'], batch_size=B)

        # NOTE: Do not back-propagate through R, t
        #       This is used to compat with the original implementation
        with torch.no_grad():
            R_loss = self.rot_loss(data)
            t_loss = self.trans_loss(data)

        loss = self_loss

        return R_loss, t_loss, loss

    def _solve_pose(self, data):
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
            batch_pts1 = uv_grid.view(B, 2, -1).transpose(1, 2) # (B, N, 2)
            batch_K1 = data['K_color1'].view(B, 3, 3)
            for idx in range(B):
                xyz_0 = batch_xyz_0[idx].cpu().numpy()
                pts1 = batch_pts1[idx].cpu().numpy()
                K1 = batch_K1[idx].cpu().numpy()

                R, t, n_inlier = self.pose_solver.estimate_pose(xyz_0, pts1, K1)

                R = torch.from_numpy(R)
                t = torch.from_numpy(t).view(1, 3)
                Rs[idx] = R
                ts[idx] = t
                n_inliers[idx] = n_inlier
                
        return Rs, ts

    def _set_single_uv_grid(self, data):
        assert 'out_HW' in data, '"out_HW" is not in data'

        B, imD, imH, imW = data['image0'].shape
        outH, outW = data['out_HW']

        is_updated = False
        if self.single_uv_grid is None:
            xs = torch.linspace(0, imW - 1, outW)
            ys = torch.linspace(0, imH - 1, outH)
            uv_grid = torch.stack(torch.meshgrid(ys, xs), dim=0).float()
            uv_grid = uv_grid.unsqueeze(0).to(data['image0'].device)
            self.single_uv_grid = uv_grid
            is_updated = True

        assert self.single_uv_grid.shape == (1, 2, outH, outW)

        return is_updated
    
    def project_3dto2d(self, K_B33, xyz_B3N, DEPTH_MIN=0.1):
        assert K_B33.shape[0] == xyz_B3N.shape[0], "K_B33.shape[0] != pts_3D_B3N.shape[0]"
        assert xyz_B3N.shape[1] == 3, "pts_3D_B3N.shape[1] != 3"
        assert K_B33.shape[1:] == (3, 3), "K_B33.shape[1:] != (3, 3)"

        # Avoid division by zero.
        # Note: negative values are also clamped at +self.options.depth_min.
        # FIXME: I do not know the unit of depth (seems to be meter), if setting too big, the predicted pixel would be wrong.
        # TODO: [THINK ABOUT THIS] In self-projection, GT would not have negative depth
        # DEPTH_MIN = 0.1

        uv_B3N = torch.bmm(K_B33, xyz_B3N)
        # Dehomogenize
        uv_B3N = uv_B3N / uv_B3N[:, 2:3, :].clamp_(min=DEPTH_MIN)
        uv_B2N = uv_B3N[:, :2, :]
        return uv_B2N