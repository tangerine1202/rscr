import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as VF

class AceBackbone(pl.LightningModule):
    """
    The feature extractor backbone used in the ACE model.

    FCN encoder, used to extract features from the input images.
    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self, cfg_encoder):
        super(AceBackbone, self).__init__()

        # NOTE: use 512 to load ACE pre-trained weight, o.w. train from scratch
        self.num_out_layers = getattr(cfg_encoder, 'NUM_OUT_LAYERS', 512)

        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1)

        self.res1_conv1 = nn.Conv2d(256, 256, 3, 1, 1)
        self.res1_conv2 = nn.Conv2d(256, 256, 1, 1, 0)
        self.res1_conv3 = nn.Conv2d(256, 256, 3, 1, 1)

        self.res2_conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.res2_conv2 = nn.Conv2d(512, 512, 1, 1, 0)
        self.res2_conv3 = nn.Conv2d(512, self.num_out_layers, 3, 1, 1)

        self.res2_skip = nn.Conv2d(256, self.num_out_layers, 1, 1, 0)


    def forward(self, x):

        # FIXME: move to dataset and control from config
        if x.shape[1] == 3:
            x = VF.rgb_to_grayscale(x)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        res = F.relu(self.conv4(x))

        x = F.relu(self.res1_conv1(res))
        x = F.relu(self.res1_conv2(x))
        x = F.relu(self.res1_conv3(x))

        res = res + x

        x = F.relu(self.res2_conv1(res))
        x = F.relu(self.res2_conv2(x))
        x = F.relu(self.res2_conv3(x))

        x = self.res2_skip(res) + x

        return x
