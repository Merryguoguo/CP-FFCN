import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np

import sys
sys.path.append('/home/guoyujun/unet/model_zoo')
from ffc_2 import FFC, FFC_BN_ACT
from ffc_resnet import BasicBlock, Bottleneck, FFCResnetBlock, FFCResNetGenerator, FFCNLayerDiscriminator
import sys
sys.path.append('/home/guoyujun/unet/util/utils')
# from SAM import SAM, SARB
from SAM_ffc import SAM, SARB, SAM_2, SARB_2
# SARB：yes LFU; SARB_2: no LFU

import pdb


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        self.input_conv.apply(weights_init('kaiming'))
        # nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, input, mask):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes, 1.0)

        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes, 0.0)

        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes, 0.0)

        return output, new_mask


class FFC_SAM_ffc_UNet_0(nn.Module):
# class FFC_SARB_UNet(nn.Module):
    # 用SARB的Attention部分
    # def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
    def __init__(self, layer_size=5, input_channels=4, output_channels=3, upsampling_mode='nearest', dilation=1, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, padding_type='reflect', attention=1):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        # ---------------------------SAM----------------------------
        self.sarb = SARB()
        # ---------------------------Encoder----------------------------
        # self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_1 = FFC_BN_ACT(input_channels, 64, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=nn.Identity,
                                activation_layer=activation_layer,
                                padding_type=padding_type, stride=2,
                                ratio_gin=0, ratio_gout=0.75,
                                enable_lfu = False)
        # self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_2 = FFC_BN_ACT(64, 128, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        padding_type=padding_type, stride=2,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)
        # self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_3 = FFC_BN_ACT(128, 256, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer, stride=2,
                        padding_type=padding_type,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)

        # ---------------------------BottleNeck----------------------------
        for i in range(3, self.layer_size): # layer_size=5
            name = 'enc_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512, 512, sample='down-3'))
            setattr(self, name, FFC_BN_ACT(256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=activation_layer, stride=2,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # 有Skip connection
        for i in range(3, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
            setattr(self, name, FFC_BN_ACT(256+256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # ---------------------------Decoder----------------------------
        # self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_3 = FFC_BN_ACT(256 + 128, 128, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_2 = FFC_BN_ACT(128 + 64, 64, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_1 = PCBActiv(64 + input_channels, input_channels, bn=False, activ=None, conv_bias=True)
        self.dec_1 = FFC_BN_ACT(64, output_channels, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=nn.Identity,
                                            activation_layer=nn.Identity,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False, bias=True)
        # pdb.set_trace()


    def forward(self, input):

        # print('----------------SARB, Generating cloud mask-----------------')
        # attention, mask = getattr(self, 'sarb')(input)
        mask = getattr(self, 'sarb')(input)
        mask_3 = torch.cat((mask, mask, mask), 1)

        h_dict = {} 
        h_dict['h_0'] = torch.cat((input, mask), 1), 0 
        h_key_prev = 'h_0'

        self.featuremap1 = mask.detach() 
        # self.featuremap2 = mask_3.detach() 

        # print('--------------------------Encoder-------------------------')
        for i in range(1, self.layer_size + 1): 
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])        
            h_key_prev = h_key

        # print('------------------------------------------------------------')
        h_key = 'h_{:d}'.format(self.layer_size) # h_5
        h_l = h_dict[h_key][0]
        h_g = h_dict[h_key][1]

        # self.featuremap1 = h_l.detach()
        # self.featuremap2 = h_g.detach()

        # print('--------------------------Decoder-------------------------')
        for i in range(self.layer_size, 0, -1):
 
            enc_h_key = 'h_{:d}'.format(i-1) 
            dec_l_key = 'dec_{:d}'.format(i)
            
            # Upsample
            h_l = F.interpolate(h_l, scale_factor=2, mode=self.upsampling_mode) 
            h_g = F.interpolate(h_g, scale_factor=2, mode=self.upsampling_mode)
            # Skip connection
            if i > 1:
                h_enc_l = torch.cat([h_l, h_dict[enc_h_key][0]], dim=1)
                h_enc_g = torch.cat([h_g, h_dict[enc_h_key][1]], dim=1)
            if i == 1:
                h_enc_l = h_l
                h_enc_g = h_g 
            h = (h_enc_l, h_enc_g)
            # Downsample
            h_dec = getattr(self, dec_l_key)(h) # h_5  
            h_l = h_dec[0]
            h_g = h_dec[1]
            h = torch.cat([h_dec[0], h_dec[1]],dim=1) 

        # return h, mask
        return h, mask_3


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class FFC_SAM_ffc_UNet(nn.Module): # yes LFU
    # def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
    def __init__(self, layer_size=5, input_channels=4, output_channels=3, upsampling_mode='nearest', dilation=1, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, padding_type='reflect'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        # ---------------------------SAM----------------------------
        self.sarb = SARB()
        # ---------------------------Encoder----------------------------
        # self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_1 = FFC_BN_ACT(input_channels, 64, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=nn.Identity,
                                activation_layer=activation_layer,
                                padding_type=padding_type, stride=2,
                                ratio_gin=0, ratio_gout=0.75,
                                enable_lfu = False)
        # self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_2 = FFC_BN_ACT(64, 128, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        padding_type=padding_type, stride=2,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)
        # self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_3 = FFC_BN_ACT(128, 256, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer, stride=2,
                        padding_type=padding_type,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)

        # ---------------------------BottleNeck----------------------------
        for i in range(3, self.layer_size): # layer_size=5
            name = 'enc_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512, 512, sample='down-3'))
            setattr(self, name, FFC_BN_ACT(256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=activation_layer, stride=2,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # 有Skip connection
        for i in range(3, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
            setattr(self, name, FFC_BN_ACT(256+256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # ---------------------------Decoder----------------------------
        # self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_3 = FFC_BN_ACT(256 + 128, 128, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_2 = FFC_BN_ACT(128 + 64, 64, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_1 = PCBActiv(64 + input_channels, input_channels, bn=False, activ=None, conv_bias=True)
        self.dec_1 = FFC_BN_ACT(64, output_channels, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=nn.Identity,
                                            activation_layer=nn.Identity,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False, bias=True)
        # pdb.set_trace()


    def forward(self, input):

        # print('----------------SARB, Generating cloud mask-----------------')
        mask = getattr(self, 'sarb')(input)
        mask_3 = torch.cat((mask, mask, mask), 1)

        h_dict = {} 
        h_dict['h_0'] = torch.cat((input, mask), 1), 0 
        h_key_prev = 'h_0'

        self.featuremap1 = mask.detach() 

        # print('--------------------------Encoder-------------------------')
        for i in range(1, self.layer_size + 1): 
            l_key = 'enc_{:d}'.format(i) # enc_1
            h_key = 'h_{:d}'.format(i) # h_
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev]) # tuple, 2           
            h_key_prev = h_key

        # print('------------------------------------------------------------')
        h_key = 'h_{:d}'.format(self.layer_size) # h_5
        h_l = h_dict[h_key][0]
        h_g = h_dict[h_key][1]

        # self.featuremap1 = h_l.detach()
        # self.featuremap2 = h_g.detach()

        # print('--------------------------Decoder-------------------------')
        for i in range(self.layer_size, 0, -1):
 
            enc_h_key = 'h_{:d}'.format(i-1) # h_4 
            dec_l_key = 'dec_{:d}'.format(i) # dec_5

            # Upsample
            h_l = F.interpolate(h_l, scale_factor=2, mode=self.upsampling_mode) # torch.Size([16, 64, 16, 16])
            h_g = F.interpolate(h_g, scale_factor=2, mode=self.upsampling_mode) # torch.Size([16, 192, 16, 16]) 
            # Skip connection
            if i > 1:
                h_enc_l = torch.cat([h_l, h_dict[enc_h_key][0]], dim=1) # torch.Size([16, 128, 16, 16])
                h_enc_g = torch.cat([h_g, h_dict[enc_h_key][1]], dim=1) # torch.Size([16, 384, 16, 16])
            if i == 1:
                h_enc_l = h_l
                h_enc_g = h_g 
            h = (h_enc_l, h_enc_g)
            # Downsample
            h_dec = getattr(self, dec_l_key)(h) # h_5  
            h_l = h_dec[0]
            h_g = h_dec[1]
            h = torch.cat([h_dec[0], h_dec[1]],dim=1) 

        return h, mask_3


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class FFC_SAM_ffc_UNet_2(nn.Module): # no LFU
    # def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
    def __init__(self, layer_size=5, input_channels=4, output_channels=3, upsampling_mode='nearest', dilation=1, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, padding_type='reflect'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        # ---------------------------SAM----------------------------
        self.sarb = SARB_2()
        # ---------------------------Encoder----------------------------
        # self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_1 = FFC_BN_ACT(input_channels, 64, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=nn.Identity,
                                activation_layer=activation_layer,
                                padding_type=padding_type, stride=2,
                                ratio_gin=0, ratio_gout=0.75,
                                enable_lfu = False)
        # self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_2 = FFC_BN_ACT(64, 128, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        padding_type=padding_type, stride=2,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)
        # self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_3 = FFC_BN_ACT(128, 256, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer, stride=2,
                        padding_type=padding_type,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)

        # ---------------------------BottleNeck----------------------------
        for i in range(3, self.layer_size): # layer_size=5
            name = 'enc_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512, 512, sample='down-3'))
            setattr(self, name, FFC_BN_ACT(256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=activation_layer, stride=2,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # 有Skip connection
        for i in range(3, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
            setattr(self, name, FFC_BN_ACT(256+256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # ---------------------------Decoder----------------------------
        # self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_3 = FFC_BN_ACT(256 + 128, 128, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_2 = FFC_BN_ACT(128 + 64, 64, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_1 = PCBActiv(64 + input_channels, input_channels, bn=False, activ=None, conv_bias=True)
        self.dec_1 = FFC_BN_ACT(64, output_channels, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=nn.Identity,
                                            activation_layer=nn.Identity,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False, bias=True)
        # pdb.set_trace()


    def forward(self, input):

        # print('----------------SARB, Generating cloud mask-----------------')
        mask = getattr(self, 'sarb')(input)
        mask_3 = torch.cat((mask, mask, mask), 1)

        h_dict = {} 
        h_dict['h_0'] = torch.cat((input, mask), 1), 0 
        h_key_prev = 'h_0'

        self.featuremap1 = mask.detach() 

        # print('--------------------------Encoder-------------------------')
        for i in range(1, self.layer_size + 1): 
            l_key = 'enc_{:d}'.format(i) # enc_1
            h_key = 'h_{:d}'.format(i) # h_
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev]) # tuple, 2           
            h_key_prev = h_key

        # print('------------------------------------------------------------')
        h_key = 'h_{:d}'.format(self.layer_size) # h_5
        h_l = h_dict[h_key][0]
        h_g = h_dict[h_key][1]

        # self.featuremap1 = h_l.detach()
        # self.featuremap2 = h_g.detach()

        # print('--------------------------Decoder-------------------------')
        for i in range(self.layer_size, 0, -1):
 
            enc_h_key = 'h_{:d}'.format(i-1) # h_4
            dec_l_key = 'dec_{:d}'.format(i) # dec_5
            
            # Upsample
            h_l = F.interpolate(h_l, scale_factor=2, mode=self.upsampling_mode) # torch.Size([16, 64, 16, 16])
            h_g = F.interpolate(h_g, scale_factor=2, mode=self.upsampling_mode) # torch.Size([16, 192, 16, 16]) 
            # Skip connection
            if i > 1:
                h_enc_l = torch.cat([h_l, h_dict[enc_h_key][0]], dim=1) # torch.Size([16, 128, 16, 16])
                h_enc_g = torch.cat([h_g, h_dict[enc_h_key][1]], dim=1) # torch.Size([16, 384, 16, 16])
            if i == 1:
                h_enc_l = h_l
                h_enc_g = h_g 
            h = (h_enc_l, h_enc_g)
            # Downsample
            h_dec = getattr(self, dec_l_key)(h) # h_5  
            h_l = h_dec[0]
            h_g = h_dec[1]
            h = torch.cat([h_dec[0], h_dec[1]],dim=1) 

        return h, mask_3


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class FFC_SAM_ffc_UNet_2_pre(nn.Module): # no LFU
    def __init__(self, layer_size=5, input_channels=4, output_channels=3, upsampling_mode='nearest', dilation=1, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, padding_type='reflect', attention=1):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        # ---------------------------SAM----------------------------
        self.sarb = SARB_2() # no LFU in Attention_ffc and SAM_ffc
        # ---------------------------Encoder----------------------------
        # self.enc_1 = PCBActiv(input_channels, 64, bn=False, sample='down-7')
        self.enc_1 = FFC_BN_ACT(input_channels, 64, kernel_size=3, padding=1, dilation=dilation,
                                norm_layer=nn.Identity,
                                activation_layer=activation_layer,
                                padding_type=padding_type, stride=2,
                                ratio_gin=0, ratio_gout=0.75,
                                enable_lfu = False)
        # self.enc_2 = PCBActiv(64, 128, sample='down-5')
        self.enc_2 = FFC_BN_ACT(64, 128, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer,
                        padding_type=padding_type, stride=2,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)
        # self.enc_3 = PCBActiv(128, 256, sample='down-5')
        self.enc_3 = FFC_BN_ACT(128, 256, kernel_size=3, padding=1, dilation=dilation,
                        norm_layer=norm_layer,
                        activation_layer=activation_layer, stride=2,
                        padding_type=padding_type,
                        ratio_gin=0.75, ratio_gout=0.75,
                        enable_lfu = False)

        # ---------------------------BottleNeck----------------------------
        for i in range(3, self.layer_size): # layer_size=5
            name = 'enc_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512, 512, sample='down-3'))
            setattr(self, name, FFC_BN_ACT(256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=activation_layer, stride=2,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # 有Skip connection
        for i in range(3, self.layer_size):
            name = 'dec_{:d}'.format(i + 1)
            # setattr(self, name, PCBActiv(512 + 512, 512, activ='leaky'))
            setattr(self, name, FFC_BN_ACT(256+256, 256, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False))

        # ---------------------------Decoder----------------------------
        # self.dec_3 = PCBActiv(256 + 128, 128, activ='leaky')
        self.dec_3 = FFC_BN_ACT(256 + 128, 128, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_2 = PCBActiv(128 + 64, 64, activ='leaky')
        self.dec_2 = FFC_BN_ACT(128 + 64, 64, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=norm_layer,
                                            activation_layer=nn.LeakyReLU,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False)
        # self.dec_1 = PCBActiv(64 + input_channels, input_channels, bn=False, activ=None, conv_bias=True)
        self.dec_1 = FFC_BN_ACT(64, output_channels, kernel_size=3, padding=1, dilation=dilation,
                                            norm_layer=nn.Identity,
                                            activation_layer=nn.Identity,
                                            padding_type=padding_type,
                                            ratio_gin=0.75, ratio_gout=0.75,
                                            enable_lfu = False, bias=True)
        # pdb.set_trace()


    def forward(self, input):

        # print('----------------SARB, Generating cloud mask-----------------')
        # attention, mask = getattr(self, 'sarb')(input)
        mask = getattr(self, 'sarb')(input)
        mask_3 = torch.cat((mask, mask, mask), 1)

        h_dict = {} 
        h_dict['h_0'] = torch.cat((input, mask), 1), 0 
        h_key_prev = 'h_0'

        self.featuremap1 = mask.detach() 
        # self.featuremap2 = mask_3.detach() 

        # print('--------------------------Encoder-------------------------')
        for i in range(1, self.layer_size + 1): 
            l_key = 'enc_{:d}'.format(i)
            h_key = 'h_{:d}'.format(i)
            h_dict[h_key] = getattr(self, l_key)(h_dict[h_key_prev])        
            h_key_prev = h_key

        # print('------------------------------------------------------------')
        h_key = 'h_{:d}'.format(self.layer_size) # h_5
        h_l = h_dict[h_key][0]
        h_g = h_dict[h_key][1]

        # self.featuremap1 = h_l.detach()
        # self.featuremap2 = h_g.detach()

        # print('--------------------------Decoder-------------------------')
        for i in range(self.layer_size, 0, -1):
 
            enc_h_key = 'h_{:d}'.format(i-1) 
            dec_l_key = 'dec_{:d}'.format(i)
            
            # Upsample
            h_l = F.interpolate(h_l, scale_factor=2, mode=self.upsampling_mode) 
            h_g = F.interpolate(h_g, scale_factor=2, mode=self.upsampling_mode)
            # Skip connection
            if i > 1:
                h_enc_l = torch.cat([h_l, h_dict[enc_h_key][0]], dim=1)
                h_enc_g = torch.cat([h_g, h_dict[enc_h_key][1]], dim=1)
            if i == 1:
                h_enc_l = h_l
                h_enc_g = h_g 
            h = (h_enc_l, h_enc_g)
            # Downsample
            h_dec = getattr(self, dec_l_key)(h) # h_5  
            h_l = h_dec[0]
            h_g = h_dec[1]
            h = torch.cat([h_dec[0], h_dec[1]],dim=1) 

        # return h, mask
        return h, mask_3


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()


class FFC_SAM_ffc_cloud(nn.Module): # yes LFU
    # def __init__(self, layer_size=7, input_channels=3, upsampling_mode='nearest'):
    def __init__(self, layer_size=5, input_channels=4, output_channels=3, upsampling_mode='nearest', dilation=1, norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU, padding_type='reflect'):
        super().__init__()
        self.freeze_enc_bn = False
        self.upsampling_mode = upsampling_mode
        self.layer_size = layer_size
        # ---------------------------SAM----------------------------
        self.sarb = SARB()

    def forward(self, input):

        # print('----------------SARB, Generating cloud mask-----------------')
        mask = getattr(self, 'sarb')(input)
        mask_3 = torch.cat((mask, mask, mask), 1)

        h_dict = {} 
        h_dict['h_0'] = torch.cat((input, mask), 1), 0 
        h_key_prev = 'h_0'

        self.featuremap1 = mask.detach() 

        return mask_3


    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super().train(mode)
        if self.freeze_enc_bn:
            for name, module in self.named_modules():
                if isinstance(module, nn.BatchNorm2d) and 'enc' in name:
                    module.eval()