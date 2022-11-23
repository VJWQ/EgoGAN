import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
import slowfast.models.optimizer as optim
import os
from .build import G_REGISTRY, D_REGISTRY


@G_REGISTRY.register()
class generator(nn.Module):
    def __init__(self, cfg):
        super(generator, self).__init__()
        print('generator init...')
        self.conv = nn.Sequential(
            nn.Conv3d(2048, 1024, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(1024),
            nn.ReLU(True),
            nn.Conv3d(1024, 512, kernel_size=1, stride=1, padding=0),
            nn.SyncBatchNorm(512),
            nn.ReLU(True),
            nn.Conv3d(512, 2, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


@D_REGISTRY.register()
class discriminator(nn.Module):
    def __init__(self, cfg):
        super(discriminator, self).__init__()
        print('discriminator init...')
        self.conv = nn.Sequential(
            nn.Conv3d(2, 32, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.SyncBatchNorm(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=1, padding=0),
            nn.SyncBatchNorm(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=0),
            nn.SyncBatchNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(128, 1, bias=True)
        self.act_func=nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = self.act_func(x)
        return x
