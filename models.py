"""
    This module contains models definition
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from activations import FTSwishPlus

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


class CenterBlock(nn.Module):
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


class CenterBlock2(nn.Module):
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


class CenterBlock3(nn.Module):
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


class RekNetM1(nn.Module):
    """
        Simple baseline FCN model:  Enc1(3->32) --> Enc2(32->32) --> Enc3(32->64) --> Enc4(64->64) --> Enc5(64->128) --> Enc6(128->256)
                                            --> center(256->256) --> 
                                    Dec6(256->128) --> Dec5(128->64) --> Dec4(64->64) --> Dec3(64->32) --> Dec2(32->32) --> Dec1(32->16) --> final(16->1)
    """
    def __init__(self, num_classes=1, ebn_enable=True, dbn_enable=True, upsample_enable=False, act_type="relu", init_type="He"):
        """
            params:
                ebn_enable      : encoder batch norm
                dbn_enable      : decoder batch norm
                upsample_enable : nn.Upsample used if this parameter is True, else nn.ConvTranspose2D used.
                act_type        : type of activation. Can be Relu, Celu or FTSwish+
                init_type       : type of initialization: He or Xavier
        """
        super(RekNetM1, self).__init__()
        self.num_classes = num_classes
        self.ebn_enable = ebn_enable
        self.dbn_enable = dbn_enable
        self.upsample_enable = upsample_enable
        self.act_type = act_type
        self.init_type = init_type


        if self.init_type not in ["He", "Xavier"]:
            raise ValueError("Unknown initialization type: {}".format(self.init_type))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_block1 = EncoderBlockM1(3, 32, bn_enable=self.ebn_enable, act_type=self.act_type)
        self.encoder_block2 = EncoderBlockM1(32, 32, bn_enable=self.ebn_enable, act_type=self.act_type)
        self.encoder_block3 = EncoderBlockM1(32, 64, bn_enable=self.ebn_enable, act_type=self.act_type)
        self.encoder_block4 = EncoderBlockM1(64, 64, bn_enable=self.ebn_enable, act_type=self.act_type)
        self.encoder_block5 = EncoderBlockM1(64, 128, bn_enable=self.ebn_enable, act_type=self.act_type)
        self.encoder_block6 = EncoderBlockM1(128, 256, bn_enable=self.ebn_enable, act_type=self.act_type)

        self.center = EncoderBlockM1(256, 256, bn_enable=self.ebn_enable, act_type=self.act_type)

        self.decoder_block6 = DecoderBlockM1(256, 128, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type)
        self.decoder_block5 = DecoderBlockM1(128, 64, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type)
        self.decoder_block4 = DecoderBlockM1(64, 64, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type)
        self.decoder_block3 = DecoderBlockM1(64, 32, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type)
        self.decoder_block2 = DecoderBlockM1(32, 32, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type)
        self.decoder_block1 = DecoderBlockM1(32, 16, bn_enable=False, upsample=self.upsample_enable, act_enable=False)
        self.final = nn.Conv2d(16, self.num_classes, kernel_size=3, padding=1)

        #Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")

    def forward(self, x):
        conv1 = self.encoder_block1(x)      
        pool1 = self.pool(conv1)
        conv2 = self.encoder_block2(pool1)
        pool2 = self.pool(conv2)  
        conv3 = self.encoder_block3(pool2)  
        pool3 = self.pool(conv3)
        conv4 = self.encoder_block4(pool3)  
        pool4 = self.pool(conv4)
        conv5 = self.encoder_block5(pool4)
        pool5 = self.pool(conv5)
        conv6 = self.encoder_block6(pool5)  
        pool6 = self.pool(conv6)

        cent = self.center(pool6)

        dec6 = self.decoder_block6(cent)
        dec5 = self.decoder_block5(dec6)
        dec4 = self.decoder_block4(dec5)
        dec3 = self.decoder_block3(dec4)
        dec2 = self.decoder_block2(dec3)
        dec1 = self.decoder_block1(dec2)

        x_out = self.final(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(x_out, dim=1)

        return x_out


class RekNetM2(nn.Module):
    def __init__(self, num_classes=1,
            ebn_enable=True, 
            dbn_enable=True, 
            upsample_enable=False,
            attention=True,
            use_skip=True,
            act_type="relu", 
            init_type="He"):
        """
            params:
                num_classes     : 
                ebn_enable      : encoder batch norm
                dbn_enable      : decoder batch norm
                upsample_enable : nn.Upsample used if this parameter is True, else nn.ConvTranspose2D used.
                attention       : CBAM attention
                use_skip        : skip-connection in Context Module
                act_type        : type of activation. Can be Relu, Celu or FTSwish+
                init_type       : type of initialization: He or Xavier
        """
        super(RekNetM2, self).__init__()
        self.num_classes = num_classes
        self.ebn_enable = ebn_enable
        self.dbn_enable = dbn_enable
        self.upsample_enable = upsample_enable
        self.attention = attention
        self.use_skip = use_skip
        self.act_type = act_type
        self.init_type = init_type

        if self.init_type not in ["He", "Xavier"]:
            raise ValueError("Unknown initialization type: {}".format(self.init_type))

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder_block1 = nn.Sequential(
            EncoderBlockM1(3, 32, bn_enable=self.ebn_enable, act_type=self.act_type),
            EncoderBlockM1(32, 32, bn_enable=self.ebn_enable, act_type=self.act_type)                                   
        )
        
        self.encoder_block2 = nn.Sequential(
            EncoderBlockM1(32, 64, bn_enable=self.ebn_enable, act_type=self.act_type),
            EncoderBlockM1(64, 64, bn_enable=self.ebn_enable, act_type=self.act_type)                                
        )

        self.encoder_block3 = nn.Sequential(
            EncoderBlockM1(64, 128, bn_enable=self.ebn_enable, act_type=self.act_type),
            EncoderBlockM1(128, 128, bn_enable=self.ebn_enable, act_type=self.act_type)                                
        )
        
        self.encoder_block4 = nn.Sequential(
            EncoderBlockM1(128, 128, bn_enable=self.ebn_enable, act_type=self.act_type),
            EncoderBlockM1(128, 128, bn_enable=self.ebn_enable, act_type=self.act_type)        
        )
        
        self.encoder_block5 = nn.Sequential(
            EncoderBlockM1(128, 256, bn_enable=self.ebn_enable, act_type=self.act_type),
            EncoderBlockM1(256, 256, bn_enable=self.ebn_enable, act_type=self.act_type)     
        )
        
        # self.center = CenterBlock(num_channels=256, act_type=self.act_type)                                                    

        self.center = nn.Sequential(
            CenterBlock3(in_channels=256, settings=[(3,1,2),(3,1,3),(3,1,4),(3,1,5)], attention=self.attention, use_skip=self.use_skip, act_type=self.act_type),
            nn.Dropout2d(p=.25)
        )

        self.decoder_block5 = nn.Sequential(
            DecoderBlockM1(256 + (512 if self.use_skip else 256), 128, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type),
            ConvBNAct(128, 128, act_type=self.act_type)
        )
        self.decoder_block4 = nn.Sequential(
            DecoderBlockM1(128 + 128, 128, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type),
            ConvBNAct(128, 128, act_type=self.act_type)
        )
        self.decoder_block3 = nn.Sequential(
            DecoderBlockM1(128 + 128, 64, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type),
            ConvBNAct(64, 64, act_type=self.act_type)
        )
        self.decoder_block2 = nn.Sequential(
            DecoderBlockM1(64 + 64, 32, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type),
            ConvBNAct(32, 32, act_type=self.act_type)
        )
        self.decoder_block1 = nn.Sequential(
            DecoderBlockM1(32 + 32, 16, bn_enable=self.dbn_enable, upsample=self.upsample_enable, act_type=self.act_type),
            ConvBNAct(16, 16, act_type=self.act_type)
        )
        self.final = nn.Conv2d(16, self.num_classes, kernel_size=3, padding=1)

        #Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_() 
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")


    def forward(self, x):
        conv1 = self.encoder_block1(x)
        pool1 = self.pool(conv1)
        conv2 = self.encoder_block2(pool1)
        pool2 = self.pool(conv2)
        conv3 = self.encoder_block3(pool2)
        pool3 = self.pool(conv3)
        conv4 = self.encoder_block4(pool3)
        pool4 = self.pool(conv4)
        conv5 = self.encoder_block5(pool4)
        pool5 = self.pool(conv5)

        cent = self.center(pool5)

        dec5 = self.decoder_block5(torch.cat([cent, pool5], dim=1))
        dec4 = self.decoder_block4(torch.cat([dec5, pool4], dim=1))
        dec3 = self.decoder_block3(torch.cat([dec4, pool3], dim=1))
        dec2 = self.decoder_block2(torch.cat([dec3, pool2], dim=1))
        dec1 = self.decoder_block1(torch.cat([dec2, pool1], dim=1))

        x_out = self.final(dec1)

        if self.num_classes > 1:
            x_out = F.log_softmax(x_out, dim=1)

        return x_out


def ConvBlock(in_ch, out_ch, ks=3, stride=1, padding=1, dilation=1, bn_enable=True):
    block = [nn.Conv2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=padding, dilation=dilation)]
    if bn_enable:
        block.append(nn.BatchNorm2d(out_ch))
    block.append(nn.ELU(inplace=True))
    return nn.Sequential(*block)


def DeconvBlock(in_ch, out_ch, ks=4, stride=2, bn_enable=True, conv_enable=True):
    block = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=1)]
    if bn_enable:
        block.append(nn.BatchNorm2d(out_ch))
    block.append(nn.ELU(inplace=True))
    if conv_enable:
        block.append(ConvBlock(out_ch, out_ch, bn_enable=bn_enable))
    return nn.Sequential(*block)


class LidCamNet(nn.Module):
    """
        This is implementaion of FCN from https://arxiv.org/pdf/1809.07941.pdf
    """
    def __init__(self, num_classes=1, input_channels=32, bn_enable=False, init_type="He"):
        super(LidCamNet, self).__init__()

        self.num_classes = num_classes
        self.input_channels = input_channels
        self.bn_enable = bn_enable
        self.init_type = init_type

        self.encoder = nn.Sequential(
            ConvBlock(3, self.input_channels, ks=4, stride=2, bn_enable=self.bn_enable),                                #1/2
            ConvBlock(self.input_channels, self.input_channels, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels, self.input_channels * 2, ks=4, stride=2, bn_enable=self.bn_enable),          #1/4
            ConvBlock(self.input_channels * 2, self.input_channels * 2, bn_enable=self.bn_enable),
            ConvBlock(self.input_channels * 2, self.input_channels * 4, ks=4, stride=2, bn_enable=self.bn_enable),      #1/8
        )

        self.center = nn.Sequential(
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),                                    #6
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),                                    #7
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(1,2), padding=(1,2), bn_enable=self.bn_enable),     #8
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(2,4), padding=(2,4), bn_enable=self.bn_enable),     #9
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(4,8), padding=(4,8), bn_enable=self.bn_enable),     #10
            nn.Dropout2d(0.25), 
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(8,16), padding=(8,16), bn_enable=self.bn_enable),   #11
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, dilation=(16,32), padding=(16,32), bn_enable=self.bn_enable), #12
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=3, bn_enable=self.bn_enable),                                    #13
            nn.Dropout2d(0.25),
            ConvBlock(self.input_channels * 4, self.input_channels * 4, ks=1, padding=0, bn_enable=self.bn_enable),                         #14
            nn.Dropout2d(0.25),
        )

        self.decoder = nn.Sequential(
            DeconvBlock(self.input_channels * 4, self.input_channels * 2, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels * 2, self.input_channels, bn_enable=self.bn_enable),
            DeconvBlock(self.input_channels, 8, bn_enable=self.bn_enable, conv_enable=False),
            nn.Conv2d(8, self.num_classes, kernel_size=3, padding=1)
        )

        #Initialization
        if self.init_type == "He":
            for m in self.modules():
                if isinstance(m, (nn.Conv2d)):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        m.bias.data.zero_() 
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
        elif self.init_type == "Xavier":
            raise NotImplementedError("This type of initialization is not implemented.")

    def forward(self, x):
        x = self.encoder(x)
        x = self.center(x)
        x = self.decoder(x)

        return x