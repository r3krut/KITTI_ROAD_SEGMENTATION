"""
    This module contains implementation of RekNetM1 - baseline FCN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules import EncoderBlockM1, DecoderBlockM1

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
