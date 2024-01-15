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
        B = data['image0'].shape[0]
        vol0 = self.encoder(data['image0'])
        vol1 = self.encoder(data['image1'])
        volume_q1k0 = self.aggregator(vol1, vol0)
        out = self.head(volume_q1k0, data)

        data['cross_xyz'] = out['cross_xyz']
        data['self_uv'] = out['self_uv']
        data['cross_uv'] = out['cross_uv']
        data['vol_HW'] = volume_q1k0.shape[-2:]

        self.set_single_uv_grid(data)
        uv_grid = self.single_uv_grid.expand(B, -1, -1, -1)

        # FIXME: use the Map-free implementation of PnP solver 
        data['R'] = torch.empty(B, 3, 3).to(data['image0'].device)
        data['t'] = torch.empty(B, 1, 3).to(data['image0'].device)
        data['inliers'] = torch.empty(B, 1).to(data['image0'].device)

        with torch.no_grad():
            batch_xyz_0 = data['cross_xyz'].view(B, 3, -1).transpose(1, 2) # (B, N, 3)
            batch_pts1 = uv_grid.view(B, 2, -1).transpose(1, 2) # (B, N, 2)
            batch_K1 = data['K_color1'].view(B, 3, 3)
            for idx in range(B):
                xyz_0 = batch_xyz_0[idx].cpu().numpy()
                pts1 = batch_pts1[idx].cpu().numpy()
                K1 = batch_K1[idx].cpu().numpy()

                R, t, inliers = self.pose_solver.estimate_pose(xyz_0, pts1, K1)

                R = torch.from_numpy(R)
                t = torch.from_numpy(t).view(1, 3)
                data['R'][idx] = R
                data['t'][idx] = t
                data['inliers'][idx] = inliers
                
        R = data['R']
        t = data['t']
        return R, t

    def loss_fn(self, data):
        B = data['image0'].shape[0]

        # FIXME: integrated into loss @data_wrapper
        self.set_single_uv_grid(data)
        uv_grid = self.single_uv_grid.expand(B, -1, -1, -1)
        self_loss, info = self.self_repro_loss(data, uv_grid)
        # cross_loss = self.cross_repro_loss(data)

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

    def set_single_uv_grid(self, data):
        assert 'vol_HW' in data, '"vol_HW" is not in data'

        B, imD, imH, imW = data['image0'].shape
        volH, volW = data['vol_HW']

        is_updated = False
        if self.single_uv_grid is None:
            xs = torch.linspace(0, imW - 1, volW)
            ys = torch.linspace(0, imH - 1, volH)
            uv_grid = torch.stack(torch.meshgrid(ys, xs), dim=0).float()
            uv_grid = uv_grid.unsqueeze(0).to(data['image0'].device)
            self.single_uv_grid = uv_grid
            is_updated = True

        assert self.single_uv_grid.shape == (1, 2, volH, volW)

        return is_updated