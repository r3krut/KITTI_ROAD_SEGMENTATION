"""
    This module contains models definition
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.bn_enable:
            x = self.bn(x)
        x = self.activation(x)
        return x

class DecoderBlockM1(nn.Module):
    """
        Base decoder block: Deconv(Upsample or TrasposeConv) -> BatchNorm(optional) -> ACTIVATION(Relu)
    """
    def __init__(self, in_channels, 
                       out_channels,
                       ks=4,
                       stride=2,
                       padding=1, 
                       bn_enable=False,
                       upsample=True):
        super(DecoderBlockM1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.bn_enable = bn_enable
        self.upsample = upsample

        self.deconv = nn.Upsample(scale_factor=2, mode='nearest') if self.upsample else nn.ConvTranspose2d(in_channels=self.in_channels,
                                                                                                           out_channels=self.out_channels,
                                                                                                           kernel_size=self.ks,
                                                                                                           stride=self.stride,
                                                                                                           padding=self.padding)

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.activation = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.out_channels)

    def forward(self, x):
        x = self.deconv(x)
        if self.upsample:
            x = self.conv(x)

        if self.bn_enable:
            x = self.bn(x)

        x = self.activation(x)
        return x


class RekNetM1(nn.Module):
    """
        Simple baseline FCN model:  Enc1(3->32) --> Enc2(32->64) --> Enc3(64->128) --> Enc4(128->256) --> Enc5(256->512) 
                                            --> center(512->512) --> 
                                    Dec5(512->256) --> Dec4(256->128) --> Dec3(128->64) --> Dec2(64->32) --> Dec1(32->1)
    """
    def __init__(self, num_classes=1, bn_enable=False):
        super(RekNetM1, self).__init__()
        self.num_classes = num_classes
        self.bn_enable = bn_enable

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder1 = EncoderBlockM1(3, 32, bn_enable=self.bn_enable)
        self.encoder2 = EncoderBlockM1(32, 64, bn_enable=self.bn_enable)
        self.encoder3 = EncoderBlockM1(64, 128, bn_enable=self.bn_enable)
        self.encoder4 = EncoderBlockM1(128, 256, bn_enable=self.bn_enable)
        self.encoder5 = EncoderBlockM1(256, 512, bn_enable=self.bn_enable)
        
        self.center = EncoderBlockM1(512, 512, bn_enable=self.bn_enable)

        self.decoder5 = DecoderBlockM1(512, 256, bn_enable=self.bn_enable)
        self.decoder4 = DecoderBlockM1(256, 128, bn_enable=self.bn_enable)
        self.decoder3 = DecoderBlockM1(128, 64, bn_enable=self.bn_enable)
        self.decoder2 = DecoderBlockM1(64, 32, bn_enable=self.bn_enable)
        self.decoder1 = DecoderBlockM1(32, self.num_classes, bn_enable=self.bn_enable)

    def forward(self, x):
        x = self.encoder1(x)
        x = self.pool(x)
        x = self.encoder2(x)
        x = self.pool(x)
        x = self.encoder3(x)
        x = self.pool(x)
        x = self.encoder4(x)
        x = self.pool(x)
        x = self.encoder5(x)
        x = self.pool(x)

        x = self.center(x)

        x = self.decoder5(x)
        x = self.decoder4(x)
        x = self.decoder3(x)
        x = self.decoder2(x)
        x = self.decoder1(x)

        return x
