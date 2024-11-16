import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from torchreid.utils.torchtools import init_params
from torchreid.components.attention import get_attention_module_instance
import numpy as np

def cachu_x_y(input):
    output = []
    height = input.size(2)
    weight = input.size(3)

    X_Y = []
    for n in range(input.size(0)):
        y = input[n][0]
        y = y.detach().cpu().numpy()
        idx = np.argwhere(y == y.max())

        x_c = idx[0][0]
        y_c = idx[0][1]
        x_y = (x_c, y_c)
        X_Y.append(x_y)

    return X_Y

def cachu(input, xy_list, h):
    output = []
    height = input.size(2)
    weight = input.size(3)

    for n, idx in enumerate(xy_list):

        x_c = idx[0]
        y_c = idx[1]
        N = torch.ones_like(input[0].unsqueeze(0))

        x1 = x_c - h
        x2 = x_c + h
        y1 = y_c - h
        y2 = y_c + h

        if x1 < 0:
            x1 = 0
        if x2 > height:
            x2 = height
        if y1 < 0:
            y1 = 0
        if y2 > weight:
            y2 = weight

        N[:, :, x1:x2, y1:y2] = 0
        N_ = N.detach().cpu().numpy()
        out = (input[n].unsqueeze(0)) * N
        output.append(out)

    out_ = torch.cat(output, 0)
    return out_


class New_Branch(nn.Module):

    def __init__(self, owner, backbone, args, input_dim):
        super().__init__()

        self.owner = weakref.ref(owner)

        self.input_dim = input_dim
        self.output_dim = args['global_dim']
        self.args = args
        self.num_classes = owner.num_classes

        self._init_fc_layer()
        if args['global_max_pooling']:
            self.avgpool = nn.AdaptiveMaxPool2d(1)
        else:
            self.avgpool = nn.AdaptiveAvgPool2d(1)
        self._init_classifier()

    def backbone_modules(self):

        return []

    def _init_classifier(self):

        classifier = nn.Linear(self.output_dim, self.num_classes)
        init_params(classifier)

        self.classifier = classifier

    def _init_fc_layer(self):

        dropout_p = self.args['dropout']

        if dropout_p is not None:
            dropout_layer = [nn.Dropout(p=dropout_p)]
        else:
            dropout_layer = []

        fc = nn.Sequential(
            nn.Linear(self.input_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.ReLU(inplace=True),
            *dropout_layer
        )
        init_params(fc)

        self.fc = fc

    def forward(self, x):

        triplet, xent, predict = [], [], []

        features_map = x

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        triplet.append(x)
        predict.append(x)
        x = self.classifier(x)
        xent.append(x)

        return predict, xent, triplet, {}, features_map

