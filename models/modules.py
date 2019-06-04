"""
    This module contains implementations of some different auxiliary modules
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.activations import FTSwishPlus

class ChannelAttention(nn.Module):
    """
        The channel attention module for CBAM - https://arxiv.org/pdf/1807.06521.pdf
        Code was taken from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    def __init__(self, in_planes, ratio=16, act_type="relu"):
        super(ChannelAttention, self).__init__()
        self.ratio = ratio
        self.act_type = act_type
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // self.ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // self.ratio, in_planes, 1, bias=False)

        if self.act_type == "relu":
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == "celu":
            self.act = nn.CELU(inplace=True)
        elif self.act_type == "fts+":
            self.act = FTSwishPlus()
        else:
            raise ValueError("Unknown value: {}".format(self.act_type))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.act(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.act(self.fc1(self.max_pool(x))))
        out = avg_out + max_out

        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
        The spatial attention module for CBAM - https://arxiv.org/pdf/1807.06521.pdf
        Code was taken from: https://github.com/luuuyi/CBAM.PyTorch/blob/master/model/resnet_cbam.py
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class ConvBNAct(nn.Module):
    """
        Simple convolution block: CONV(ks=3) -> BN(optional) -> ACTIVATION(optional)
    """
    def __init__(self, in_channels, 
                out_channels,
                ks=3,
                stride=1,
                padding=1,
                dilation=1, 
                bn_enable=True, 
                act_enable=True,
                act_type="relu"):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bn_enable = bn_enable
        self.act_enable = act_enable
        self.act_type = act_type

        assert self.act_type in ["relu", "celu", "fts+"], "Error. Unknown activation function: {}".format(self.act_type)

        self.conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=self.ks, stride=self.stride, padding=self.padding, dilation=self.dilation)
        
        if self.bn_enable:
            self.bn = nn.BatchNorm2d(self.out_channels)
        
        if self.act_enable:
            if self.act_type == "relu":
                self.act = nn.ReLU(inplace=True)
            elif self.act_type == "celu":
                self.act = nn.CELU(inplace=True)
            elif self.act_type == "fts+":
                self.act = FTSwishPlus()
            else:
                raise ValueError("Unknown value: {}".format(self.act_type))

    def forward(self, x):
        x = self.conv(x)
        
        if self.bn_enable:
            x = self.bn(x)

        if self.act_enable:
            x = self.act(x)

        return x


class EncoderBlockM1(nn.Module):
    """
        Base encoder block: CONV -> BatchNorm(optional) -> ACTIVATION(RELU, CELU or FTS+)
    """
    def __init__(self, in_channels, 
                       out_channels,
                       ks=3,
                       stride=1,
                       padding=1,
                       dilation=1, 
                       bn_enable=False,
                       act_type="relu"):
        super(EncoderBlockM1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bn_enable = bn_enable
        self.act_type = act_type

        self.conv = ConvBNAct(self.in_channels, self.out_channels, 
            ks=self.ks, 
            stride=self.stride, 
            padding=self.padding, 
            dilation=self.dilation, 
            bn_enable=self.bn_enable, 
            act_type=self.act_type)

    def forward(self, x):
        x = self.conv(x)
        return x


class DecoderBlockM1(nn.Module):
    """
        Base decoder block: Deconv(Upsample or TrasposeConv2D) -> BatchNorm(optional) -> ACTIVATION(Relu, Celu or FTS+)(Optional)
    """
    def __init__(self, in_channels,
                       out_channels,
                       ks=4,
                       stride=2,
                       padding=1, 
                       dilation=1,
                       bn_enable=False,
                       upsample=False,
                       act_enable=True,
                       act_type="relu"):
        super(DecoderBlockM1, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bn_enable = bn_enable
        self.upsample = upsample
        self.act_enable = act_enable
        self.act_type = act_type

        assert self.act_type in ["relu", "celu", "fts+"], "Error. Unknown activation function: {}".format(self.act_type)

        self.deconv = nn.Upsample(scale_factor=2, mode='bilinear') if self.upsample else nn.ConvTranspose2d(in_channels=self.in_channels,
                                                                                                           out_channels=self.out_channels,
                                                                                                           kernel_size=self.ks,
                                                                                                           stride=self.stride,
                                                                                                           padding=self.padding,
                                                                                                           dilation=self.dilation)

        if self.upsample:
            self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)

        if self.bn_enable:
            self.bn = nn.BatchNorm2d(self.out_channels)

        if self.act_enable:
            if self.act_type == "relu":
                self.act = nn.ReLU(inplace=True)
            elif self.act_type == "celu":
                self.act = nn.CELU(inplace=True)
            elif self.act_type == "fts+":
                self.act = FTSwishPlus()
            else:
                raise ValueError("Unknown value: {}".format(self.act_type))

    def forward(self, x):
        x = self.deconv(x)
        if self.upsample:
            x = self.conv(x)

        if self.bn_enable:
            x = self.bn(x)

        if self.act_enable:
            x = self.act(x)

        return x


class DecoderBlockM2(nn.Module):
    """
        Decoder block with middle channels: 
            1: Upsample -> ConvBNAct -> ConvBNAct
            2: ConvBNAct -> ConvTranspose -> Act
    """
    def __init__(self, in_channels,
                       middle_channels, 
                       out_channels,
                       ks=4,
                       stride=2,
                       padding=1, 
                       dilation=1,
                       upsample=False,
                       act_type="relu"):
        """
            params:
                in_channels         :
                middle_channels     :
                out_channels        :
                ks                  : kernel size
                stride              :
                padding             :
                dilation            :
                upsample            : Uses nn.Upsample if True else nn.ConvTranspose2D
        """
        super(DecoderBlockM2, self).__init__()
        
        self.in_channels = in_channels
        self.mid_channels = middle_channels
        self.out_channels = out_channels
        self.ks = ks
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.upsample = upsample
        self.act_type = act_type

        if self.upsample:
            self.block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear"),
                ConvBNAct(in_channels=self.in_channels, out_channels=self.mid_channels, act_type=self.act_type),
                ConvBNAct(in_channels=self.mid_channels, out_channels=self.out_channels, act_type=self.act_type)
            )
        else:
            self.block = nn.Sequential(
                ConvBNAct(in_channels=self.in_channels, out_channels=self.mid_channels, act_type=self.act_type),
                nn.ConvTranspose2d(in_channels=self.mid_channels, out_channels=self.out_channels, kernel_size=self.ks, stride=self.stride, padding=self.padding, dilation=self.dilation),
                nn.BatchNorm2d(self.out_channels)
            )

        if not self.upsample:
            if self.act_type == "relu":
                self.act = nn.ReLU(inplace=True)
            elif self.act_type == "celu":
                self.act = nn.CELU(inplace=True)
            elif self.act_type == "fts+":
                self.act = FTSwishPlus()
            else:
                raise ValueError("Unknown value: {}".format(self.act_type))

    def forward(self, x):
        x = self.block(x)

        if not self.upsample:
            x = self.act(x)
        
        return x


class CenterBlockM1(nn.Module):
    """
        Center module. This module consists from sequential convolutions and drop-outs modules.
    """

    def __init__(self, num_channels=32, act_type="relu"):
        super().__init__()
        self.num_channels = num_channels
        self.act_type = act_type

        self.block = nn.Sequential(
            ConvBNAct(self.num_channels, self.num_channels, ks=3, padding=(1,2), dilation=(1,2), act_type=self.act_type), #256
            nn.Dropout2d(p=.25),  
            ConvBNAct(self.num_channels, self.num_channels, ks=3, padding=(1,3), dilation=(1,3), act_type=self.act_type), #256
            nn.Dropout2d(p=.25),
            ConvBNAct(self.num_channels, self.num_channels, ks=3, padding=(1,4), dilation=(1,4), act_type=self.act_type), #256
            nn.Dropout2d(p=.25),
            ConvBNAct(self.num_channels, self.num_channels, ks=3, padding=(1,5), dilation=(1,5), act_type=self.act_type), #256
            nn.Dropout2d(p=.25),
            ConvBNAct(self.num_channels, self.num_channels, ks=1, padding=(0,0), dilation=(1,1), act_type=self.act_type), #256
            nn.Dropout2d(p=.25)
        )

    def forward(self, x):
        x = self.block(x)
        return x


class CenterBlockM2(nn.Module):
    """
        This center block is like as Pyramid Scene Parsing module in PSPNet with a several modifications. Original paper https://arxiv.org/pdf/1612.01105.pdf
    """
    def __init__(self, in_channels=256, settings=(1,2,3,6), pooling_type="max", bn_enable=True, act_type="relu"):
        super().__init__()

        assert len(settings) > 0, "Wrong settings: {}".format(settings)
        assert pooling_type in ["max", "avg"], "Unknown pooling: {}".format(pooling_type)
        assert act_type in ["relu", "celu", "fts+"], "Unknown activation: {}".format(act_type)

        self.in_channels = in_channels
        self.settings = settings
        self.out_channels = self.in_channels // len(self.settings)
        self.pooling_type = pooling_type
        self.bn_enable = bn_enable
        self.act_type = act_type

        self.pyramid = []
        for s in self.settings:
            pyramid_block = []
            if self.pooling_type == "max":
                pyramid_block.append(nn.AdaptiveMaxPool2d(s))
            else:
                pyramid_block.append(nn.AdaptiveAvgPool2d(s))
            
            pyramid_block.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1))

            if self.bn_enable:
                pyramid_block.append(nn.BatchNorm2d(num_features=self.out_channels))
            
            if self.act_type == "relu":
                pyramid_block.append(nn.ReLU(inplace=True))
            elif self.act_type == "celu":
                pyramid_block.append(nn.CELU(inplace=True))
            else:
                pyramid_block.append(FTSwishPlus())
            self.pyramid.append(nn.Sequential(*pyramid_block))
        
        self.pyramid = nn.ModuleList(self.pyramid)


    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]
        
        result = [x]
        for p in self.pyramid:
            res_x = p(x)
            res_x = F.upsample(res_x, size=(h,w), mode="bilinear")
            result.append(res_x)
        
        return torch.cat(result, dim=1)


class CenterBlockM3(nn.Module):
    """
        This is PSP-like module with custom modifications
    """
    def __init__(self, in_channels=256, settings=[(3,1,1),(3,2,2)], fusion_type="cat", act_type="relu", attention=False, use_skip=True):
        """
            Params:
                in_channels     : input channels
                settings        : (kernel_size, dilation_h, dilation_w)   
                fustion_type    : type of fustion. Can be "cat" or "sum"
                act_type        : activation type
                attention       : CBAM attention enabling
                use_skip        : if True then source tensor will be reductioned with others, else not.
        """
        super().__init__()

        assert len(settings) % 2 == 0, "Wrong number of levels in pyramid: {}".format(len(settings))

        self.act_type = act_type
        self.settings = settings
        self.fusion_type = fusion_type
        self.attention = attention
        self.use_skip = use_skip
        self.in_channels = in_channels
        self.out_channels = self.in_channels // len(self.settings) 

        if self.attention:
            self.ca = ChannelAttention(self.out_channels, act_type=self.act_type)
            self.sa = SpatialAttention(kernel_size=3)

        self.pyramid = []
        for s in self.settings:
            ks, d_h, d_w = s
            self.pyramid.append(ConvBNAct(in_channels=self.in_channels, 
                    out_channels=self.out_channels,
                    ks=ks,
                    padding=((ks//2) * d_h, (ks//2) * d_w),
                    dilation=(d_h, d_w),
                    act_type=self.act_type 
                )
            )
        self.pyramid = nn.ModuleList(self.pyramid)

    def forward(self, x):
        if self.fusion_type == "cat":
            result = [x] if self.use_skip else []
            for p in self.pyramid:
                res_x = p(x)
                
                if self.attention:
                    res_x = self.ca(res_x) * res_x
                    res_x = self.sa(res_x) * res_x

                result.append(res_x) 
            return torch.cat(result, dim=1)         #concatination of features
        elif self.fusion_type == "sum":
            result = []
            for p in self.pyramid:
                res_x = p(x)

                if self.attention:
                    res_x = self.ca(res_x) * res_x
                    res_x = self.sa(res_x) * res_x

                result.append(res_x)
            result = torch.cat(result, dim=1)      
            return x + result                       #sum of features
        else:
            raise RuntimeError("Unknown fusion type: {}".format(self.fusion_type))