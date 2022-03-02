# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule, build_norm_layer, build_activation_layer
from collections import OrderedDict

from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

from mmseg.models.dynamic_conv import dsc

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        dsc_cfg = kwargs.pop('dsc', None)
        fuse_dsc_cfg = kwargs.pop('fuse_dsc', None)
        
        super(SegFormerHead, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']

        self.with_dsc = dsc_cfg is not None
        self.fuse_with_dsc = fuse_dsc_cfg is not None
        
        if self.with_dsc:
            self.linear_c4 = dsc(
                cfg=dsc_cfg,
                in_planes=c4_in_channels, 
                out_planes=embedding_dim, 
                kernel_size=3,  
                stride=1, 
                bias=True,
                init_weight=True, 
            )

            self.linear_c3 = dsc(
                cfg=dsc_cfg,
                in_planes=c3_in_channels, 
                out_planes=embedding_dim, 
                kernel_size=3,  
                stride=1, 
                bias=True,
                init_weight=True, 
            )

            self.linear_c2 = dsc(
                cfg=dsc_cfg,
                in_planes=c2_in_channels, 
                out_planes=embedding_dim, 
                kernel_size=3,  
                stride=1, 
                bias=True,
                init_weight=True, 
            )

            self.linear_c1 = dsc(
                cfg=dsc_cfg,
                in_planes=c1_in_channels, 
                out_planes=embedding_dim, 
                kernel_size=3,  
                stride=1, 
                bias=True,
                init_weight=True, 
            )
        
        else:
            self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
            self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
            self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
            self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        if self.fuse_with_dsc:
            self.linear_fuse = nn.Sequential(
                dsc(
                    cfg=fuse_dsc_cfg,
                    in_planes=embedding_dim*4, 
                    out_planes=embedding_dim, 
                    kernel_size=3,  
                    stride=1, 
                    bias=True,
                    init_weight=True, 
                ),
                build_norm_layer(dict(type='SyncBN', requires_grad=True), embedding_dim)[1],
                build_activation_layer(dict(type='ReLU', inplace=True)),
            )
        else:
            self.linear_fuse = ConvModule(
                in_channels=embedding_dim*4,
                out_channels=embedding_dim,
                kernel_size=1,
                norm_cfg=dict(type='SyncBN', requires_grad=True)
            )
        
        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        if self.with_dsc:
            _c4 = self.linear_c4(c4)
            _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c3 = self.linear_c3(c3)
            _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c2 = self.linear_c2(c2)
            _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c1 = self.linear_c1(c1)
        
        else:
            _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
            _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
            _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
            _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

            _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        
        return x
