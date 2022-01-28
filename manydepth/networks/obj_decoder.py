# Copyright Niantic 2021. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the ManyDepth licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from distutils.command.build import build
import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
import datetime
from manydepth.layers import ConvBlock, Conv3x3, upsample
from MonoFlex.model.head.detector_head import bulid_head

from MonoFlex.config import cfg
cfg.merge_from_file('MonoFlex/runs/monoflex.yaml')

BN_MOMENTUM = 0.1


class OBJDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=1, use_skips=True):
        super(OBJDecoder, self).__init__()
        self.inplanes = 512
        num_ch_enc_obj = 128
        self.deconv_with_bias = False
        # used for deconv layers
        self.deconv_layers = self._make_deconv_layer(
            3,
            [256, 256, 256],
            [4, 4, 4],
        )
        self.transfer = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.heads = bulid_head(cfg, num_ch_enc_obj)

        # depth decoder
        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, index):
        if deconv_kernel == 4:
            padding = 1
            output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def forward(self, input_features, targets = None, is_test = False, training_phase = 'depth'):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        if training_phase == 'object':
            obj_branch_x = self.deconv_layers(x)
            obj_branch_x = self.transfer(obj_branch_x)
            self.outputs["mono3d"] = self.heads(obj_branch_x, targets, is_test)

        else:
            for i in range(4, -1, -1):
                x = self.convs[("upconv", i, 0)](x)
                x = [upsample(x)]
                if self.use_skips and i > 0:
                    x += [input_features[i - 1]]
                x = torch.cat(x, 1)
                x = self.convs[("upconv", i, 1)](x)
                if i in self.scales:
                    self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        
        return self.outputs
