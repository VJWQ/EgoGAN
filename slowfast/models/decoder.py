import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import slowfast.models.optimizer as optim
import os
import slowfast.utils.weight_init_helper as init_helper
from slowfast.models.batchnorm_helper import get_norm

from . import head_helper, resnet_helper, stem_helper
from .build import DE_REGISTRY

@DE_REGISTRY.register()
class decoder(nn.Module):
    """
    ResNet model builder. It builds a ResNet like network backbone without
    lateral connection (C2D, I3D, Slow).
    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    Xiaolong Wang, Ross Girshick, Abhinav Gupta, and Kaiming He.
    "Non-local neural networks."
    https://arxiv.org/pdf/1711.07971.pdf
    """
    def __init__(self, cfg):
        super(decoder, self).__init__()
        print('decoder init...')
        self.norm_module = get_norm(cfg)
        self.enable_detection = cfg.DETECTION.ENABLE
        self.num_pathways = 1
        self._construct_network(cfg)
        init_helper.init_weights(
            self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        )

    def _construct_network(self, cfg):
        self.flow_conv = nn.Conv3d(2, 32, kernel_size=1)
        self.relu    = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose3d(2050, 1024, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
        self.bn1     = nn.BatchNorm3d(1024)
        self.deconv2 = nn.ConvTranspose3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), output_padding=(0, 1, 1))
        self.bn2     = nn.BatchNorm3d(512)
        self.deconv3 = nn.ConvTranspose3d(512, 256, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
        self.bn3     = nn.BatchNorm3d(256)
        self.deconv4 = nn.ConvTranspose3d(256, 64, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 0, 0))
        self.bn4     = nn.BatchNorm3d(64)
        self.deconv5 = nn.ConvTranspose3d(64, 64, kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1), dilation=(2, 1, 1), output_padding=(0, 1, 1))
        self.bn5     = nn.BatchNorm3d(64)
        self.deconv6 = nn.Conv3d(64, 32, kernel_size=(4, 1, 1))
        self.bn6 = nn.BatchNorm3d(32)
        self.deconv7= nn.Conv3d(32, 16, kernel_size=(3, 1, 1))
        self.bn7 = nn.BatchNorm3d(16)
        self.classifier = nn.Conv3d(16, 2, kernel_size=1)

    def forward(self, x, bboxes=None):
        x1, x2, x3, x4, x5, optflow = x[0], x[1], x[2], x[3], x[4], x[5]
        score = torch.cat((optflow, x5[0]), dim=1)  # torch.Size([8, 2050, 4, 7, 7])
        score = self.bn1(self.relu(self.deconv1(score)))    # torch.Size([1, 1024, 4, 14, 14])      x4
        score = score + x4[0]
        score = self.bn2(self.relu(self.deconv2(score)))    # torch.Size([1, 512, 4, 28, 28])       x3
        score = score + x3[0]
        score = self.bn3(self.relu(self.deconv3(score)))    # torch.Size([1, 256, 4, 56, 56])       x2
        score = score + x2[0]
        score = self.bn4(self.relu(self.deconv4(score)))    # torch.Size([1, 64, 8, 56, 56])        x1
        score = score + x1[0]
        score = self.bn5(self.relu(self.deconv5(score)))  # torch.Size([1, 64, 8, 224, 224])
        score = self.bn6(self.relu(self.deconv6(score)))
        score = self.bn7(self.relu(self.deconv7(score)))  # torch.Size([1, 16, 3, 224, 224])
        score = self.classifier(score)  # torch.Size([1, 2, 3, 224, 224])

        return score
