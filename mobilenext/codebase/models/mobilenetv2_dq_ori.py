"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018). 
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch
import torch.nn as nn
import math

from torch.nn.parameter import Parameter
import torch.nn.functional as F

#this version use mobilenetv2 model withoyt sequential packing. trying to use dual 3x3 layer with more shortcut connections
__all__ = ['mobilenetv2_dq_ori']


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, rm_1x1 = True, interpolation = False, group_1x1=False, force_1x1 = False, shuffle=False,
                 coeff_mom=0.9):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.expand_ratio = expand_ratio
        self.identity = stride == 1 and inp == oup
        self.rm_1x1 = rm_1x1
        self.interpolation = interpolation
        self.inp, self.oup = inp, oup
        self.high_dim_id = False

        if self.expand_ratio != 1:
            self.conv_exp = nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False)
            self.bn1 = nn.BatchNorm2d(hidden_dim)
        # self.depth_sep_conv = nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False)
        # self.bn2 = nn.BatchNorm2d(hidden_dim)
        self.depth_sep_conv_1 = nn.Conv2d(hidden_dim, hidden_dim, (3,3), (stride,stride), (1,1), groups=hidden_dim, bias=False)
        self.depth_sep_conv_2 = nn.Conv2d(hidden_dim, hidden_dim, (3,3), (stride,stride), (1,1), groups=hidden_dim, bias=False)
        self.bn2_1 = nn.BatchNorm2d(hidden_dim)
        self.bn2_2 = nn.BatchNorm2d(hidden_dim)

        # self.depth_sep_conv = nn.Conv2d(hidden_dim, hidden_dim, (3,3), (stride,stride), (1,1), groups=hidden_dim, bias=False)
        # self.bn2 = nn.BatchNorm2d(hidden_dim)

        self.conv_pro = nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

        self.relu = nn.ReLU6(inplace=True)
        self.max_ref_1 = nn.MaxPool2d(kernel_size=3, stride=1, dilation=2, padding=0)
        self.max_ref_2 = nn.MaxPool2d(kernel_size=3, stride=1, dilation=3, padding=0)
    def cha_shuffle(self,x, coeff, group_num = 2, fix_coeff=False):
        """
            use interpolation to shuffle the channels.
        """
        base_len = x.shape[1]//group_num
        if fix_coeff:
            x_new_1 = 0.5 * x[:,:base_len,:,:] + 0.5 * x[:,base_len:,:,:]
            x_new_2 = 0.5 * x[:,:base_len,:,:] + 0.5 * x[:,base_len:,:,:]
        else:
            x_new_1 = coeff[0] * x[:,:base_len,:,:] + (1 - coeff[0]) * x[:,base_len:,:,:]
            x_new_2 = coeff[1] * x[:,:base_len,:,:] + (1 - coeff[1]) * x[:,base_len:,:,:]
        x_new = torch.cat([x_new_1, x_new_2], dim = 1)
        return x_new
    def forward(self, input):
        if isinstance(input,tuple):
            shortcut = input[1]
            x = input[0]
            self.high_dim_id = True
            # import pdb;pdb.set_trace()
        else:
            x= input
        if self.expand_ratio !=1:
            x = self.relu(self.bn1(self.conv_exp(x)))
        x = self.relu(self.bn2_1(self.depth_sep_conv_1(x)))
        # x = self.bn2_2(self.depth_sep_conv_2(x))
        x = self.bn3(self.conv_pro(x))
        if self.identity:
            return x + input
        else:
            return x


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1., rm_1x1 = True, interpolation = False, group_1x1=False):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        # 1x1 config in each block
        force_1x1 = [False, False, False, False, False, True, True]
        idx_1x1 = 0
        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t, rm_1x1 = rm_1x1, interpolation = interpolation, 
                              group_1x1=group_1x1, force_1x1 = force_1x1[idx_1x1]))
                input_channel = output_channel
            idx_1x1 += 1
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else 1280
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def mobilenetv2_dq_ori(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

