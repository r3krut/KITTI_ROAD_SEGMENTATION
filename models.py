"""
    This module contains models definition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class EncoderBlockM1(nn.Module):
    """
        Base encoder block: CONV -> BatchNorm(optional) -> ACTIVATION(RELU)
    """
    def __init__(self, in_channels, 
                       out_channels,
                       ks=3,
                       stride=1,
                       padding=1, 
                       bn_enable=False):
        super(EncoderBlockM1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.bn_enable = bn_enable

        self.conv = nn.Conv2d(in_channels=self.in_channels, 
                            out_channels=self.out_channels,
                            kernel_size=self.ks,
                            stride=self.stride,
                            padding=self.padding)
        
        self.activation = nn.ReLU(inplace=True)

        if self.bn_enable:
            self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn_enable:
            x = self.bn(x)
        x = self.activation(x)
        return x


class DecoderBlockM1(nn.Module):
    """
        Base decoder block: Deconv(Upsample or TrasposeConv2D) -> BatchNorm(optional) -> ACTIVATION(Relu)(Optional)
    """
    def __init__(self, in_channels, 
                       out_channels,
                       ks=4,
                       stride=2,
                       padding=1, 
                       bn_enable=False,
                       upsample=False,
                       act_enable=True):
        super(DecoderBlockM1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.bn_enable = bn_enable
        self.upsample = upsample
        self.act_enable=act_enable

        self.deconv = nn.Upsample(scale_factor=2, mode='bilinear') if self.upsample else nn.ConvTranspose2d(in_channels=self.in_channels,
                                                                                                           out_channels=self.out_channels,
                                                                                                           kernel_size=self.ks,
                                                                                                           stride=self.stride,
                                                                                                           padding=self.padding)

        if self.upsample:
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        
        if self.act_enable:
            self.activation = nn.ReLU(inplace=True)

        if self.bn_enable:
            self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.deconv(x)
        if self.upsample:
            x = self.conv(x)

        if self.bn_enable:
            x = self.bn(x)

        if self.act_enable:
            x = self.activation(x)

        return x


class RekNetM1(nn.Module):
    """
        Simple baseline FCN model:  Enc1(3->32) --> Enc2(32->32) --> Enc3(32->64) --> Enc4(64->64) --> Enc5(64->128) 
                                            --> center(128->128) --> 
                                    Dec5(128->64) --> Dec4(64->64) --> Dec3(64->32) --> Dec2(32->32) --> Dec1(32->1)
    """
    def __init__(self, num_classes=1, ebn_enable=True, dbn_enable=True, upsample_enable=False, init_type="He"):
        """
            params:
                ebn_enable      : encoder batch norm
                dbn_enable      : decoder batch norm
                upsample_enable : nn.Upsample used if this parameter is True, else nn.ConvTranspose2D used.
                init_type       : type of initialization: He or Xavier
        """
        super(RekNetM1, self).__init__()
        self.num_classes = num_classes
        self.ebn_enable = ebn_enable
        self.dbn_enable = dbn_enable
        self.upsample_enable = upsample_enable
        self.init_type = init_type

        if self.init_type not in ["He", "Xavier"]:
            raise ValueError("Unknown initialization type: {}".format(self.init_type))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_block1 = EncoderBlockM1(3, 32, bn_enable=self.ebn_enable)
        self.encoder_block2 = EncoderBlockM1(32, 32, bn_enable=self.ebn_enable)
        self.encoder_block3 = EncoderBlockM1(32, 64, bn_enable=self.ebn_enable)
        self.encoder_block4 = EncoderBlockM1(64, 64, bn_enable=self.ebn_enable)
        self.encoder_block5 = EncoderBlockM1(64, 128, bn_enable=self.ebn_enable)
        self.encoder_block6 = EncoderBlockM1(128, 256, bn_enable=self.ebn_enable)

        # self.center = EncoderBlockM1(128, 128, bn_enable=self.ebn_enable)
        self.center = EncoderBlockM1(256, 256, bn_enable=self.ebn_enable)

        self.decoder_block6 = DecoderBlockM1(256, 128, bn_enable=self.dbn_enable, upsample=self.upsample_enable)
        self.decoder_block5 = DecoderBlockM1(128, 64, bn_enable=self.dbn_enable, upsample=self.upsample_enable)
        self.decoder_block4 = DecoderBlockM1(64, 64, bn_enable=self.dbn_enable, upsample=self.upsample_enable)
        self.decoder_block3 = DecoderBlockM1(64, 32, bn_enable=self.dbn_enable, upsample=self.upsample_enable)
        self.decoder_block2 = DecoderBlockM1(32, 32, bn_enable=self.dbn_enable, upsample=self.upsample_enable)
        self.decoder_block1 = DecoderBlockM1(32, 1, bn_enable=False, upsample=self.upsample_enable, act_enable=False)


        #Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")

    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.pool(x)
        x = self.encoder_block2(x)
        x = self.pool(x)
        x = self.encoder_block3(x)
        x = self.pool(x)
        x = self.encoder_block4(x)
        x = self.pool(x)
        x = self.encoder_block5(x)
        x = self.pool(x)
        x = self.encoder_block6(x)
        x = self.pool(x)

        x = self.center(x)

        x = self.decoder_block6(x)
        x = self.decoder_block5(x)
        x = self.decoder_block4(x)
        x = self.decoder_block3(x)
        x = self.decoder_block2(x)
        x = self.decoder_block1(x)

        return x

