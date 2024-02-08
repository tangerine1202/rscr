import math
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from kornia.geometry.conversions import quaternion_to_rotation_matrix, QuaternionCoeffOrder
from scipy.spatial.transform import Rotation

from lib.models.regression.head import DeepResBlock
from lib.models.regression.encoder.preact import PreActBlock
from lib.utils.solver import procrustes
from lib.utils.rotationutils import rotation_matrix_from_ortho6d


class Deep1x1BlockMLP(pl.LightningModule):
    def __init__(self, cfg, in_channels):
        super().__init__()
        # all 1x1 conv
        self.mlp1x1 = torch.nn.Sequential(
            *[

                torch.nn.LazyConv2d(256, 1, 1, 0, bias=True),
                torch.nn.ReLU(),
                torch.nn.Conv2d(256, 128, 1, 1, 0, bias=True),
                torch.nn.ReLU(),
                torch.nn.Conv2d(128, 64, 1, 1, 0, bias=True),
            ])

    def forward(self, feature_volume):
        x = self.mlp1x1(feature_volume)
        return x


class DirectDeepTranslationMLP(DeepResBlock):
    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.mlp = torch.nn.Sequential(
            *[
                torch.nn.LazyLinear(256, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 128, bias=True),
                torch.nn.ReLU(),
                torch.nn.Linear(128, 3, bias=True)
            ])

    def forward(self, feature_volume, data):
        B = feature_volume.shape[0]
        x = super().forward(feature_volume)
        out = self.mlp(x).view(B, 3)

        return out


class NaiveSCHead(Deep1x1BlockMLP):
    def __init__(self, cfg, in_channels):
        super().__init__(cfg, in_channels)

        self.fc = torch.nn.LazyConv2d(3, 1, 1, 0, bias=True)
    
    def forward(self, feature_volume):
        x = super().forward(feature_volume)
        x = self.fc(x)
        return x


class AceHead(pl.LightningModule):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def __init__(self,
                 cfg,
                 in_channels=512,
                 mean=torch.tensor([0.0, 0.0, 0.0]),
                 num_head_blocks=1,
                 use_homogeneous=True,
                 homogeneous_min_scale=0.01,
                 homogeneous_max_scale=4.0):
        super(AceHead, self).__init__()

        self.use_homogeneous = use_homogeneous
        self.in_channels = in_channels  # Number of encoder features.
        self.head_channels = 512  # Hardcoded.

        # We may need a skip layer if the number of features output by the encoder is different.
        # self.head_skip = torch.nn.Identity() if self.in_channels == self.head_channels else torch.nn.Conv2d(self.in_channels, self.head_channels, 1, 1, 0)
        self.head_skip = torch.nn.LazyConv2d(self.head_channels, 1, 1, 0)

        self.res3_conv1 = torch.nn.LazyConv2d(self.head_channels, 1, 1, 0)
        self.res3_conv2 = torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.res3_conv3 = torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        self.res_blocks = []

        for block in range(num_head_blocks):
            self.res_blocks.append((
                torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
                torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0),
            ))

            super(AceHead, self).add_module(str(block) + 'c0', self.res_blocks[block][0])
            super(AceHead, self).add_module(str(block) + 'c1', self.res_blocks[block][1])
            super(AceHead, self).add_module(str(block) + 'c2', self.res_blocks[block][2])

        self.fc1 = torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)
        self.fc2 = torch.nn.Conv2d(self.head_channels, self.head_channels, 1, 1, 0)

        if self.use_homogeneous:
            self.fc3 = torch.nn.Conv2d(self.head_channels, 4, 1, 1, 0)

            # Use buffers because they need to be saved in the state dict.
            self.register_buffer("max_scale", torch.tensor([homogeneous_max_scale]))
            self.register_buffer("min_scale", torch.tensor([homogeneous_min_scale]))
            self.register_buffer("max_inv_scale", 1. / self.max_scale)
            self.register_buffer("h_beta", math.log(2) / (1. - self.max_inv_scale))
            self.register_buffer("min_inv_scale", 1. / self.min_scale)
        else:
            self.fc3 = torch.nn.Conv2d(self.head_channels, 3, 1, 1, 0)

        # Learn scene coordinates relative to a mean coordinate (e.g. center of the scene).
        self.register_buffer("mean", mean.clone().detach().view(1, 3, 1, 1))

    def forward(self, res):

        x = F.relu(self.res3_conv1(res))
        x = F.relu(self.res3_conv2(x))
        x = F.relu(self.res3_conv3(x))

        res = self.head_skip(res) + x

        for res_block in self.res_blocks:
            x = F.relu(res_block[0](res))
            x = F.relu(res_block[1](x))
            x = F.relu(res_block[2](x))

            res = res + x

        sc = F.relu(self.fc1(res))
        sc = F.relu(self.fc2(sc))
        sc = self.fc3(sc)

        if self.use_homogeneous:
            # Dehomogenize coords:
            # Softplus ensures we have a smooth homogeneous parameter with a minimum value = self.max_inv_scale.
            h_slice = F.softplus(sc[:, 3, :, :].unsqueeze(1), beta=self.h_beta.item()) + self.max_inv_scale
            h_slice.clamp_(max=self.min_inv_scale.item())
            sc = sc[:, :3] / h_slice

        # Add the mean to the predicted coordinates.
        sc += self.mean

        return sc
