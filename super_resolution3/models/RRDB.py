import torch
from torch import nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn import init as init
import numpy as np
from super_resolution3.models.common import AdjustedNonLocalBlock,Channel_NonLocalBlock2D
from diffusion_model3.diffusion import Diffusion
from diffusion_model3.unet import UNetModel
from super_resolution3.utils import inference_mini_batch,inference_mini_batch_random
import torch.nn.functional as F
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class SFTLayer(nn.Module):
    def __init__(self, in_channels=32, inter_channels=64):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(in_channels, in_channels, 1)
        self.SFT_scale_conv1 = nn.Conv2d(in_channels, inter_channels, 1)
        self.SFT_shift_conv0 = nn.Conv2d(in_channels, in_channels, 1)
        self.SFT_shift_conv1 = nn.Conv2d(in_channels, inter_channels, 1)

    def forward(self, fea, cond):
        # x[0]: fea; x[1]: cond
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(cond), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(cond), 0.1, inplace=True))
        return fea * (scale+1)  + shift


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Empirically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Empirically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class feature_fuse(nn.Module):
    def __init__(self,in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,inter_channel=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel1, inter_channel, 1, 1, 0)
        self.conv2 = nn.Conv2d(in_channel2, inter_channel, 1, 1, 0)
        self.conv3 = nn.Conv2d(in_channel3, inter_channel, 1, 1, 0)
        self.conv4 = nn.Conv2d(in_channel4, inter_channel, 1, 1, 0)
        self.conv5 = nn.Conv2d(in_channel5, inter_channel, 1, 1, 0)

        self.fuse1 = AdjustedNonLocalBlock(inter_channel, inter_channel // 2)
        self.fuse2 = AdjustedNonLocalBlock(inter_channel, inter_channel // 2)
        self.fuse3 = AdjustedNonLocalBlock(inter_channel, inter_channel // 2)
        self.fuse4 = AdjustedNonLocalBlock(inter_channel, inter_channel // 2)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self,feature_list):
        fea1,fea2,fea3,fea4,fea5 = feature_list
        fea1, fea2, fea3, fea4, fea5 = fea1.to(self.device),fea2.to(self.device),fea3.to(self.device),fea4.to(self.device),fea5.to(self.device)

        fea1 = self.conv1(fea1)
        fea2 = self.conv2(fea2)
        fea3 = self.conv3(fea3)
        fea4 = self.conv4(fea4)
        fea5 = self.conv5(fea5)

        out = self.fuse1(fea2,fea1)
        out = self.fuse2(fea3,out)
        out = self.fuse3(fea4,out)
        out = self.fuse4(fea5,out)

        return out


class diff_rrdm(nn.Module):
    def __init__(self,in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,inter_channel=256,output_channel=128):
        super().__init__()
        self.fea_fuse = feature_fuse(in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,inter_channel) # 输出通道为256
        self.sr_net = RRDBNet(inter_channel + output_channel, output_channel)
    def forward(self,x,feature_list):
        out = self.fea_fuse(feature_list)
        out = torch.cat([x,out],dim=1)
        out = self.sr_net(out)
        return out


class diff_guided_rrdm(nn.Module):
    '''
    在5,10,15,20处加入特征引导
    '''
    def __init__(self, num_in_ch, num_out_ch, in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,inter_channel=256, num_feat=64, num_block=23, num_grow_ch=32):
        '''
        :param num_in_ch: 主干网络输入的通道数,即x的通道数,128
        :param num_out_ch: 128
        :param in_channel1: 6144
        :param in_channel2: 5632
        :param in_channel3: 3328
        :param in_channel4: 1664
        :param in_channel5: 896
        :param inter_channel: 特征融合的输出通道数
        :param num_feat: RRDB的输入
        :param num_block: RRDB的个数
        :param num_grow_ch: 修改RRDB内部的卷积通道
        '''
        super(diff_guided_rrdm, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList(
            [
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat + num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat + 2 * num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat + 3 * num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, num_block - 20, num_feat=num_feat + 4 * num_feat, num_grow_ch=num_grow_ch)
            ]
        )
        self.conv_body = nn.Conv2d(num_feat + 4 * num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # diffusion feature branch
        self.fea_fuse = feature_fuse(in_channel1, in_channel2, in_channel3, in_channel4, in_channel5, inter_channel)
        self.conv1_prior = nn.Conv2d(inter_channel,num_feat,3,1,1)
        self.rrdb_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            ]
        )


    def forward(self, x, feature_list):
        feat = self.conv_first(x)

        prior = self.fea_fuse(feature_list)
        prior = self.conv1_prior(prior)
        for i,layer in enumerate(self.body):
            if i == 0:
                prior = self.rrdb_conv[i](prior)
                out = layer(feat)
                out = torch.cat([out, prior], dim=1)
            elif i== 4:
                out = layer(out)
            else:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = torch.cat([out, prior], dim=1)

        body_feat = self.conv_body(out)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class diff_guided_rrdm_v2(nn.Module):
    '''
    在5,10,15,20处加入特征引导
    输入加入特征
    '''
    def __init__(self, num_in_ch, num_out_ch, in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,inter_channel=256, num_feat=64, num_block=23, num_grow_ch=32):
        '''
        :param num_in_ch: 主干网络输入的通道数,即x的通道数,128
        :param num_out_ch: 128
        :param in_channel1: 6144
        :param in_channel2: 5632
        :param in_channel3: 3328
        :param in_channel4: 1664
        :param in_channel5: 896
        :param inter_channel: 特征融合的输出通道数
        :param num_feat: RRDB的输入
        :param num_block: RRDB的个数
        :param num_grow_ch: 修改RRDB内部的卷积通道
        '''
        super(diff_guided_rrdm_v2, self).__init__()
        self.conv_first = nn.Conv2d(inter_channel + num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList(
            [
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat + num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat + 2 * num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat + 3 * num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, num_block - 20, num_feat=num_feat + 4 * num_feat, num_grow_ch=num_grow_ch)
            ]
        )
        self.conv_body = nn.Conv2d(num_feat + 4 * num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # diffusion feature branch
        self.fea_fuse = feature_fuse(in_channel1, in_channel2, in_channel3, in_channel4, in_channel5, inter_channel)
        self.conv1_prior = nn.Conv2d(inter_channel,num_feat,3,1,1)
        self.rrdb_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            ]
        )


    def forward(self, x, feature_list):
        prior = self.fea_fuse(feature_list)
        feat = self.conv_first(torch.cat([x,prior],dim=1))
        prior = self.conv1_prior(prior)
        for i,layer in enumerate(self.body):
            if i == 0:
                prior = self.rrdb_conv[i](prior)
                out = layer(feat)
                out = torch.cat([out, prior], dim=1)
            elif i== 4:
                out = layer(out)
            else:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = torch.cat([out, prior], dim=1)

        body_feat = self.conv_body(out)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class diff_guided_rrdm_v3(nn.Module):
    '''
    在5,10,15,20处加入特征引导
    输入加入特征
    修改了inter_channel = 128
    '''
    def __init__(self, num_in_ch, num_out_ch, in_channel1,in_channel2,in_channel3,in_channel4,in_channel5,inter_channel=128, num_feat=64, num_block=23, num_grow_ch=32):
        '''
        :param num_in_ch: 主干网络输入的通道数,即x的通道数,128
        :param num_out_ch: 128
        :param in_channel1: 6144
        :param in_channel2: 5632
        :param in_channel3: 3328
        :param in_channel4: 1664
        :param in_channel5: 896
        :param inter_channel: 特征融合的输出通道数
        :param num_feat: RRDB的输入
        :param num_block: RRDB的个数
        :param num_grow_ch: 修改RRDB内部的卷积通道
        '''
        super(diff_guided_rrdm_v3, self).__init__()
        self.conv_first = nn.Conv2d(inter_channel + num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList(
            [
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, num_block - 20, num_feat=num_feat, num_grow_ch=num_grow_ch)
            ]
        )

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # diffusion feature branch
        self.fea_fuse = feature_fuse(in_channel1, in_channel2, in_channel3, in_channel4, in_channel5, inter_channel)
        self.conv1_prior = nn.Conv2d(inter_channel,num_feat,3,1,1)
        self.rrdb_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            ]
        )

        # SFT融合
        self.sft = make_layer(SFTLayer, 4 ,in_channels=num_feat, inter_channels=num_feat)

    def forward(self, x, feature_list):
        prior = self.fea_fuse(feature_list)  # (_,128,16,16)
        feat = self.conv_first(torch.cat([x,prior],dim=1)) # (_,64,16,16)
        prior = self.conv1_prior(prior)  # (_,64,16,16)
        for i,layer in enumerate(self.body):
            if i == 0:
                prior = self.rrdb_conv[i](prior)
                out = layer(feat)
                out = self.sft[i](out,prior)
            elif i== 4:
                out = layer(out)
            else:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.sft[i](out,prior)

        body_feat = self.conv_body(out)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class TAM(nn.Module):
    def __init__(self,
                 in_channels=256,
                 n_segment=5,
                 kernel_size=3,
                 stride=1,
                 padding=1):
        super(TAM, self).__init__()
        self.in_channels = in_channels
        self.n_segment = n_segment
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        # print('TAM with kernel_size {}.'.format(kernel_size))

        self.G = nn.Sequential(
            nn.Linear(n_segment, n_segment * 2, bias=False),
            nn.BatchNorm1d(n_segment * 2), nn.ReLU(inplace=True),
            nn.Linear(n_segment * 2, kernel_size, bias=False), nn.Softmax(-1))

        self.L = nn.Sequential(
            nn.Conv1d(in_channels,
                      in_channels // 4,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False), nn.BatchNorm1d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // 4, in_channels, 1, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        # x.size = N*C*T*(H*W)
        nt, c, h, w = x.size()
        t = self.n_segment
        n_batch = nt // t
        new_x = x.view(n_batch, t, c, h, w).permute(0, 2, 1, 3,
                                                     4).contiguous()
        out = F.adaptive_avg_pool2d(new_x.view(n_batch * c, t, h, w), (1, 1))
        out = out.view(-1, t)
        conv_kernel = self.G(out.view(-1, t)).view(n_batch * c, 1, -1, 1)
        local_activation = self.L(out.view(n_batch, c,
                                           t)).view(n_batch, c, t, 1, 1)
        new_x = new_x * local_activation
        out = F.conv2d(new_x.view(1, n_batch * c, t, h * w),
                       conv_kernel,
                       bias=None,
                       stride=(self.stride, 1),
                       padding=(self.padding, 0),
                       groups=n_batch * c)
        out = out.view(n_batch, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).contiguous() #.view(nt, c, h, w)

        return out


class feature_fuse2(nn.Module):
    def __init__(self, inchannel = 7680, inter_channel=256):
        super().__init__()
        self.inter_channel = inter_channel
        # 对通道数降维
        self.conv = nn.Conv2d(inchannel, inter_channel,1,1,0)
        # Time Attention
        self.time_attn = TAM()

    def forward(self, fea):
        # 输入(_,10,7680,16,16)
        batch,num_t,c,h,w = fea.shape
        fea = fea.view(-1,c,h,w)
        out = self.conv(fea)
        out = out.view(-1,self.inter_channel,h,w)  #(_,20,256,16,16)
        out = self.time_attn(out)
        out = torch.sum(out,dim=1)
        return out



class diff_guided_rrdm_adaptive(nn.Module):
    '''
    在5,10,15,20处加入特征引导, 不再是并联而是自适应相加
    在RRDB后面加入非局部自适应, 通道上的
    '''
    def __init__(self, num_in_ch, num_out_ch, inter_channel=256, num_feat=64, num_block=23, num_grow_ch=32):
        '''
        :param num_in_ch: 主干网络输入的通道数,即x的通道数,128
        :param num_out_ch: 128
        :param in_channel1: 6144
        :param in_channel2: 5632
        :param in_channel3: 3328
        :param in_channel4: 1664
        :param in_channel5: 896
        :param inter_channel: 特征融合的输出通道数
        :param num_feat: RRDB的输入
        :param num_block: RRDB的个数
        :param num_grow_ch: 修改RRDB内部的卷积通道
        '''
        super(diff_guided_rrdm_adaptive, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList(
            [
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, num_block - 20, num_feat=num_feat, num_grow_ch=num_grow_ch)
            ]
        )
        # cnl Channel_NonLocalBlock2D
        self.cnl1 = Channel_NonLocalBlock2D(num_feat)
        self.cnl2 = Channel_NonLocalBlock2D(num_feat)
        self.cnl3 = Channel_NonLocalBlock2D(num_feat)
        self.cnl4 = Channel_NonLocalBlock2D(num_feat)
        self.cnl5 = Channel_NonLocalBlock2D(num_feat)

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) # 8
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # diffusion feature branch
        self.fea_fuse = feature_fuse2()
        self.conv1_prior = nn.Conv2d(inter_channel,num_feat,3,1,1)
        self.rrdb_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            ]
        )
        self.alpha1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha1.data.fill_(0.1)
        self.alpha2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha2.data.fill_(0.1)
        self.alpha3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha3.data.fill_(0.1)
        self.alpha4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha4.data.fill_(0.1)


    def forward(self, x, feature_list):
        feat = self.conv_first(x)

        prior = self.fea_fuse(feature_list)  # (_,256,16,16)
        prior = self.conv1_prior(prior)
        for i,layer in enumerate(self.body):
            if i == 0:
                prior = self.rrdb_conv[i](prior)
                out = layer(feat)
                out = self.cnl1(out)
                out = out + self.alpha1*prior
            elif i == 1:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl2(out)
                out = out + self.alpha2*prior
            elif i == 2:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl3(out)
                out = out + self.alpha3*prior
            elif i == 3:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl4(out)
                out = out + self.alpha4*prior
            else:
                out = layer(out)
                out = self.cnl5(out)

        body_feat = self.conv_body(out)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest'))) # 8
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


class diff_guided_rrdm_adaptive_x8(nn.Module):
    '''
    在5,10,15,20处加入特征引导, 不再是并联而是自适应相加
    在RRDB后面加入非局部自适应, 通道上的
    '''
    def __init__(self, num_in_ch, num_out_ch, inter_channel=256, num_feat=64, num_block=23, num_grow_ch=32):
        '''
        :param num_in_ch: 主干网络输入的通道数,即x的通道数,128
        :param num_out_ch: 128
        :param in_channel1: 6144
        :param in_channel2: 5632
        :param in_channel3: 3328
        :param in_channel4: 1664
        :param in_channel5: 896
        :param inter_channel: 特征融合的输出通道数
        :param num_feat: RRDB的输入
        :param num_block: RRDB的个数
        :param num_grow_ch: 修改RRDB内部的卷积通道
        '''
        super(diff_guided_rrdm_adaptive_x8, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList(
            [
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, num_block - 20, num_feat=num_feat, num_grow_ch=num_grow_ch)
            ]
        )
        # cnl Channel_NonLocalBlock2D
        self.cnl1 = Channel_NonLocalBlock2D(num_feat)
        self.cnl2 = Channel_NonLocalBlock2D(num_feat)
        self.cnl3 = Channel_NonLocalBlock2D(num_feat)
        self.cnl4 = Channel_NonLocalBlock2D(num_feat)
        self.cnl5 = Channel_NonLocalBlock2D(num_feat)

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) # 8
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # diffusion feature branch
        self.fea_fuse = feature_fuse2()
        self.conv1_prior = nn.Conv2d(inter_channel,num_feat,3,1,1)
        self.rrdb_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            ]
        )
        self.alpha1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha1.data.fill_(0.1)
        self.alpha2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha2.data.fill_(0.1)
        self.alpha3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha3.data.fill_(0.1)
        self.alpha4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha4.data.fill_(0.1)


    def forward(self, x, feature_list):
        feat = self.conv_first(x)

        prior = self.fea_fuse(feature_list)  # (_,256,16,16)
        prior = self.conv1_prior(prior)
        for i,layer in enumerate(self.body):
            if i == 0:
                prior = self.rrdb_conv[i](prior)
                out = layer(feat)
                out = self.cnl1(out)
                out = out + self.alpha1*prior
            elif i == 1:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl2(out)
                out = out + self.alpha2*prior
            elif i == 2:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl3(out)
                out = out + self.alpha3*prior
            elif i == 3:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl4(out)
                out = out + self.alpha4*prior
            else:
                out = layer(out)
                out = self.cnl5(out)

        body_feat = self.conv_body(out)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest'))) # 8
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# v2就是不是采用参数相加而是SFT,效果不好
class diff_guided_rrdm_adaptive_v2(nn.Module):
    '''
    在5,10,15,20处加入特征引导, 不再是并联而是自适应相加
    在RRDB后面加入非局部自适应, 通道上的
    '''
    def __init__(self, num_in_ch, num_out_ch, inter_channel=256, num_feat=64, num_block=23, num_grow_ch=32):
        '''
        :param num_in_ch: 主干网络输入的通道数,即x的通道数,128
        :param num_out_ch: 128
        :param in_channel1: 6144
        :param in_channel2: 5632
        :param in_channel3: 3328
        :param in_channel4: 1664
        :param in_channel5: 896
        :param inter_channel: 特征融合的输出通道数
        :param num_feat: RRDB的输入
        :param num_block: RRDB的个数
        :param num_grow_ch: 修改RRDB内部的卷积通道
        '''
        super(diff_guided_rrdm_adaptive_v2, self).__init__()
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.body = nn.ModuleList(
            [
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, 5, num_feat=num_feat, num_grow_ch=num_grow_ch),
                make_layer(RRDB, num_block - 20, num_feat=num_feat, num_grow_ch=num_grow_ch)
            ]
        )
        # cnl Channel_NonLocalBlock2D
        self.cnl1 = Channel_NonLocalBlock2D(num_feat)
        self.cnl2 = Channel_NonLocalBlock2D(num_feat)
        self.cnl3 = Channel_NonLocalBlock2D(num_feat)
        self.cnl4 = Channel_NonLocalBlock2D(num_feat)
        self.cnl5 = Channel_NonLocalBlock2D(num_feat)

        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1) # 8
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # diffusion feature branch
        self.fea_fuse = feature_fuse2()
        self.conv1_prior = nn.Conv2d(inter_channel,num_feat,3,1,1)
        self.rrdb_conv = nn.ModuleList(
            [
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)]),
                nn.Sequential(
                    *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch), nn.Conv2d(num_feat, num_feat, 3, 1, 1)])
            ]
        )
        self.alpha1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha1.data.fill_(0.1)
        self.alpha2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha2.data.fill_(0.1)
        self.alpha3 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha3.data.fill_(0.1)
        self.alpha4 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)  # the size is [1]
        self.alpha4.data.fill_(0.1)

        self.sft = make_layer(SFTLayer, 4, in_channels=num_feat, inter_channels=num_feat)

    def forward(self, x, feature_list):
        feat = self.conv_first(x)

        prior = self.fea_fuse(feature_list)  # (_,256,16,16)
        prior = self.conv1_prior(prior)
        for i,layer in enumerate(self.body):
            if i == 0:
                prior = self.rrdb_conv[i](prior)
                out = layer(feat)
                out = self.cnl1(out)
                out = self.sft[i](out,prior)
            elif i == 1:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl2(out)
                out = self.sft[i](out,prior)
            elif i == 2:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl3(out)
                out = self.sft[i](out,prior)
            elif i == 3:
                prior = self.rrdb_conv[i](prior)
                out = layer(out)
                out = self.cnl4(out)
                out = self.sft[i](out,prior)
            else:
                out = layer(out)
                out = self.cnl5(out)

        body_feat = self.conv_body(out)
        feat = feat + body_feat

        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest'))) # 8
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out



if __name__ == "__main__":
    # ------------------------------------------------------------------------------------------------------------- #
    # RRDB
    # net = RRDBNet(128,128)
    # x = torch.rand(100,128,16,16)
    # y = net(x)
    # print(y.shape)

    # ------------------------------------------------------------------------------------------------------------- #
    # diffusion
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNetModel(
        image_size=16,
        in_channels=128,
        model_channels=128,
        out_channels=128,
        num_res_blocks=2,
        attention_resolutions=set([4, 8]),
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        channel_mult=(1, 2, 3, 4),
        dropout=0.0,
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)

    X = torch.rand(100,128,16,16).to(device)
    feature = inference_mini_batch_random(model,X)
    # fuse_net = feature_fuse2()
    # fuse_out = fuse_net(feature)
    # print(fuse_out.shape)


    # print(feature[0].shape[1],feature[1].shape[1],feature[2].shape[1],feature[3].shape[1],feature[4].shape[1])
    # net = diff_guided_rrdm_v3(128,128,feature[0].shape[1],feature[1].shape[1],feature[2].shape[1],feature[3].shape[1],feature[4].shape[1]).to(device)
    # out = net(X,feature)
    # print(out.shape) # (100,128,64,64)

    net = diff_guided_rrdm_adaptive(128,128).to(device)
    out = net(X, feature)
    print(out.shape)

