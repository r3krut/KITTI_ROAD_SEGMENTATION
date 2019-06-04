"""
    This module contains implementation of RekNetM2
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import EncoderBlockM1, CenterBlockM3, DecoderBlockM1, ConvBNAct

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
            CenterBlockM3(in_channels=256, settings=[(3,1,2),(3,1,3),(3,1,4),(3,1,5)], attention=self.attention, use_skip=self.use_skip, act_type=self.act_type),
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