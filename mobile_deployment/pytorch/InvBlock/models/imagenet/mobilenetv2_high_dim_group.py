"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
import from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import torch.nn as nn
import math

__all__ = ['mobilenetv2_high_dim', 'edgenet']


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

def group_conv_1x1_bn(inp, oup, expand_ratio):
    hidden_dim = oup // expand_ratio
    return nn.Sequential(
        nn.Conv2d(inp, hidden_dim, 1, 1, 0, groups=hidden_dim, bias=False),
        nn.BatchNorm2d(hidden_dim),
        nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class InvertedResidualV2(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidualV2, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        #self.identity = inp == oup
        #assert inp == oup

        #self.relu = nn.ReLU6(inplace=True)
        self.expand_ratio = expand_ratio
        if expand_ratio == 1:
            self.conv1 = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            hidden_dim = inp // expand_ratio
            self.conv1 = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                #nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(inp),
            )

    def forward(self, x):
        out = self.conv1(x)
        if self.expand_ratio != 1:
            return x + out
        else:
            return out

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        hidden_dim_2 = round(inp * expand_ratio * 4)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                # nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
                # nn.BatchNorm2d(hidden_dim),
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = None
            if self.identity:
                self.conv1 = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    )
                self.conv2 = nn.Sequential(
                    # group 1x1
                    nn.Conv2d(hidden_dim, hidden_dim_2, 1, 1, 0, groups=12, bias=False),
                    nn.BatchNorm2d(hidden_dim_2),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim_2, hidden_dim_2, 3, stride, 1, groups=hidden_dim_2, bias=False),
                    nn.BatchNorm2d(hidden_dim_2),
                    nn.ReLU6(inplace=True),
                    # group 1x1
                    nn.Conv2d(hidden_dim_2, hidden_dim, 1, 1, 0, groups=12, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    )
                    #pw linear
                self.conv3 = nn.Sequential(
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim*2, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim*2),
                    nn.ReLU6(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim*2, hidden_dim*2, 3, stride, 1, groups=hidden_dim*2, bias=False),
                    nn.BatchNorm2d(hidden_dim*2),
                    nn.ReLU6(inplace=True),
                    #pw linear
                    nn.Conv2d(hidden_dim*2, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )

    def forward(self, x):
        if self.identity and self.conv is None:
                y = self.conv1(x)
                y = self.conv2(y) + y
                y = self.conv3(y)
                return y + x
        else:
            return self.conv(x)


class EdgeNet(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(EdgeNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [4,  24, 2, 2],
            [4,  36, 2, 2],
            [4,  72, 2, 2],
            [4,  96, 2, 1],
            [4, 160, 2, 2],
            [4, 320, 1, 1],
        ]
        #self.cfgs = [
        #    # t, c, n, s
        #    [1,  16, 1, 1],
        #    [4,  24, 2, 2],
        #    [4,  32, 3, 2],
        #    [4,  64, 3, 2],
        #    [4,  96, 4, 1],
        #    [4, 160, 3, 2],
        #    [4, 320, 1, 1],
        #]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidualV2
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            if t == 1:
                layers.append(block(input_channel, output_channel, s, t))
                input_channel = output_channel
                continue
            for i in range(n):
                if i == 0:
                    layers.append(BlockTransition(input_channel, output_channel * t, s))
                layers.append(block(output_channel * t, output_channel * t, s, t))
                if n > 1 and i == n - 1:
                    layers.append(BlockTransition(output_channel * t, output_channel, relu=False))
                    input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        input_channel = output_channel * 4
        output_channel = _make_divisible(input_channel * width_mult, 4 if width_mult == 0.1 else 8) if width_mult > 1.0 else input_channel
        #self.conv = conv_1x1_bn(1280, 320)
        #self.conv = group_conv_1x1_bn(1280, 1280, 4)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(output_channel, num_classes)
                )

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.conv(x)
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

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNetV2, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [3,  24, 2, 2],
            [3,  32, 3, 2],
            [3,  64, 4, 2],
            [3,  96, 3, 1],
            [3, 160, 3, 2],
            [3, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
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

def mobilenetv2_high_dim(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return MobileNetV2(**kwargs)

def edgenet(**kwargs):
    """
    Constructs a MobileNet V2 model
    """
    return EdgeNet(**kwargs)

