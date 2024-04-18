import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from submodules import Transform


class PointNet(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super(PointNet, self).__init__()
        self.global_feat = global_feat
        self.feature_transform = feature_transform

        self.in_trans = Transform.TransformNetwork(k=3)    # input transformation layer
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        if self.feature_transform:
            self.ft_trans = Transform.TransformNetwork(k=64)    # feature transformation

    def forward(self, x):
        n_pts = x.size()[2]

        trans = self.in_trans(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.ft_trans(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat