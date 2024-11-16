import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict

from torchreid.utils.torchtools import init_params
from torchreid.components.attention import get_attention_module_instance

from torchreid.components.new_module import *
from torchreid.components.OutlookAttention import OutlookAttention
from torchreid.components.CGAFusion import CGAFusion

class MultiBranchNetwork(nn.Module):

    def __init__(self, backbone, args, num_classes, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.common_branch = self._get_common_branch(backbone, args)
        self.branches = nn.ModuleList(self._get_branches(backbone, args))
        self.branches_new = New_Branch(self, backbone, args, 1024)
        self._init_attention_sup_layer()
        self.Attention = OutlookAttention(dim=1024)
        self.fusion = CGAFusion(1024)


    def _get_common_branch(self, backbone, args):
        return NotImplemented


    def _get_middle_subbranch_for(self, backbone, args, last_branch_class):

        return NotImplemented


    def _init_attention_sup_layer(self):
        self.input_dim = 2048
        self.output_dim = 1024
        r = self.input_dim / self.output_dim

        attention_sup = nn.Sequential(
            nn.Conv2d(self.input_dim, int(self.input_dim / r), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(self.input_dim / r)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.input_dim / r), int(self.input_dim / (r * r)), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(self.input_dim / (r * r))),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(self.input_dim / (r * r)), 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

        init_params(attention_sup)
        self.attention_sup_ = attention_sup

    def backbone_modules(self):

        lst = [*self.common_branch.backbone_modules()]
        for branch in self.branches:
            lst.extend(branch.backbone_modules())

        return lst

    def forward(self, x, return_featuremaps=False):
        x, *intermediate_fmaps = self.common_branch(x)
        x_new_branch = x

        fmap_dict = defaultdict(list)
        fmap_dict['intermediate'].extend(intermediate_fmaps)

        predict_features, xent_features, triplet_features = [], [], []

        for branch in self.branches:
            predict, xent, triplet, fmap, global_fmap = branch(x)
            predict_features.extend(predict)
            xent_features.extend(xent)
            triplet_features.extend(triplet)

            if len(global_fmap) > 0:
                g_fmap = global_fmap

            for name, fmap_list in fmap.items():
                fmap_dict[name].extend(fmap_list)

        fmap_dict = {k: tuple(v) for k, v in fmap_dict.items()}

        f_local = torch.cat(fmap_dict['after'], 1)
        f_gl = g_fmap

        if return_featuremaps:
            f_local = torch.cat(fmap_dict['after'], 1)
            f_gl = g_fmap
            feature = torch.cat([f_gl, f_local], dim = 2)

            return f_local, g_fmap


        x_sup_frame_after = self.attention_sup_(g_fmap)
        x_y = cachu_x_y(x_sup_frame_after)

        x_new_branch_yizhi = cachu(x_new_branch, x_y, h=3)
        x_new_branch_yizhi = x_new_branch_yizhi.permute(0, 2, 3, 1)
        x_new_branch_att = self.Attention(x_new_branch_yizhi)
        x_new_branch_att_l = x_new_branch_att.permute(0, 3, 1, 2)
        x_new_branch_att_h = x_new_branch

        x_fusion = self.fusion(x_new_branch_att_l, x_new_branch_att_h)

        predict_new_branch, xent_new_branch, triplet_new_branch, fmap_new_branch, global_fmap_new_branch = self.branches_new(x_fusion)

        predict_features.extend(predict_new_branch)
        xent_features.extend(xent_new_branch)
        triplet_features.extend(triplet_new_branch)

        return torch.cat(predict_features, 1), tuple(xent_features), tuple(triplet_features), fmap_dict


class Sequential(nn.Sequential):

    def backbone_modules(self):
        backbone_modules = []
        for m in self._modules.values():
            backbone_modules.append(m.backbone_modules())

        return backbone_modules


class GlobalBranch(nn.Module):

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
