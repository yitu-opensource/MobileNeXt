
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from .activations import sigmoid
from .conv2d_layers import *


# Defaults used for Google/Tensorflow training of mobile networks /w RMSprop as per
# papers and TF reference implementations. PT momentum equiv for TF decay is (1 - TF decay)
# NOTE: momentum varies btw .99 and .9997 depending on source
# .99 in official TF TPU impl
# .9997 (/w .999 in search space) for paper
BN_MOMENTUM_TF_DEFAULT = 1 - 0.99
BN_EPS_TF_DEFAULT = 1e-3
_BN_ARGS_TF = dict(momentum=BN_MOMENTUM_TF_DEFAULT, eps=BN_EPS_TF_DEFAULT)


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

def get_bn_args_tf():
    return _BN_ARGS_TF.copy()


def resolve_bn_args(kwargs):
    bn_args = get_bn_args_tf() if kwargs.pop('bn_tf', False) else {}
    bn_momentum = kwargs.pop('bn_momentum', None)
    if bn_momentum is not None:
        bn_args['momentum'] = bn_momentum
    bn_eps = kwargs.pop('bn_eps', None)
    if bn_eps is not None:
        bn_args['eps'] = bn_eps
    return bn_args


_SE_ARGS_DEFAULT = dict(
    gate_fn=sigmoid,
    act_layer=None,
    reduce_mid=False,
    divisor=1)


def resolve_se_args(kwargs, in_chs, act_layer=None):
    se_kwargs = kwargs.copy() if kwargs is not None else {}
    # fill in args that aren't specified with the defaults
    for k, v in _SE_ARGS_DEFAULT.items():
        se_kwargs.setdefault(k, v)
    # some models, like MobilNetV3, calculate SE reduction chs from the containing block's mid_ch instead of in_ch
    if not se_kwargs.pop('reduce_mid'):
        se_kwargs['reduced_base_chs'] = in_chs
    # act_layer override, if it remains None, the containing block's act_layer will be used
    if se_kwargs['act_layer'] is None:
        assert act_layer is not None
        se_kwargs['act_layer'] = act_layer
    return se_kwargs


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def round_channels(channels, multiplier=1.0, divisor=8, channel_min=None):
    """Round number of filters based on depth multiplier."""
    if not multiplier:
        return channels
    channels *= multiplier
    return make_divisible(channels, divisor, channel_min)


def drop_connect(inputs, training=False, drop_connect_rate=0.):
    """Apply drop connect."""
    if not training:
        return inputs

    keep_prob = 1 - drop_connect_rate
    random_tensor = keep_prob + torch.rand(
        (inputs.size()[0], 1, 1, 1), dtype=inputs.dtype, device=inputs.device)
    random_tensor.floor_()  # binarize
    output = inputs.div(keep_prob) * random_tensor
    return output


class ChannelShuffle(nn.Module):
    # FIXME haven't used yet
    def __init__(self, groups):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        """Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]"""
        N, C, H, W = x.size()
        g = self.groups
        assert C % g == 0, "Incompatible group size {} for input channel {}".format(
            g, C
        )
        return (
            x.view(N, g, int(C / g), H, W)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
            .view(N, C, H, W)
        )


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class ConvBnAct(nn.Module):
    def __init__(self, in_chs, out_chs, kernel_size,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None):
        super(ConvBnAct, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.conv = select_conv2d(in_chs, out_chs, kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn1 = norm_layer(out_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

    def feature_module(self, location):
        return 'act1'

    def feature_channels(self, location):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn1(x)
        x = self.act1(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """ DepthwiseSeparable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 pw_kernel_size=1, pw_act=False, se_ratio=0., se_kwargs=None,
                 norm_layer=nn.BatchNorm2d, norm_kwargs=None, drop_connect_rate=0.):
        super(DepthwiseSeparableConv, self).__init__()
        norm_kwargs = norm_kwargs or {}
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        self.drop_connect_rate = drop_connect_rate

        self.conv_dw = select_conv2d(
            in_chs, in_chs, dw_kernel_size, stride=stride, dilation=dilation, padding=pad_type, depthwise=True)
        self.bn1 = norm_layer(in_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(in_chs, se_ratio=se_ratio, **se_kwargs)

        self.conv_pw = select_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True) if self.has_pw_act else nn.Identity()

    def feature_module(self, location):
        # no expansion in this block, pre pw only feature extraction point
        return 'conv_pw'

    def feature_channels(self, location):
        return self.conv_pw.in_channels

    def forward(self, x):
        residual = x

        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.act1(x)

        if self.has_se:
            x = self.se(x)

        x = self.conv_pw(x)
        x = self.bn2(x)
        x = self.act2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE and CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_connect_rate=0.):
        super(InvertedResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate

        # Point-wise expansion
        self.conv_pw = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Depth-wise convolution
        self.conv_dw = select_conv2d(
            mid_chs, mid_chs, dw_kernel_size, stride=stride, dilation=dilation,
            padding=pad_type, depthwise=True, **conv_kwargs)
        self.bn2 = norm_layer(mid_chs, **norm_kwargs)
        self.act2 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_layer(out_chs, **norm_kwargs)

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_pw.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def forward(self, x):
        residual = x

        # Point-wise expansion
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        return x


# class I2RBlock(nn.Module):
#     """ Inverted residual block w/ optional SE and CondConv routing"""

#     def __init__(self, in_chs, out_chs, dw_kernel_size=3,
#                  stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
#                  exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
#                  se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
#                  conv_kwargs=None, drop_connect_rate=0., keep_3x3 = False, group_1x1=1):
#         super(I2RBlock, self).__init__()
#         norm_kwargs = norm_kwargs or {}
#         conv_kwargs = conv_kwargs or {}
#         mid_chs = make_divisible(in_chs * exp_ratio)
#         self.has_se = se_ratio is not None and se_ratio > 0.
#         self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
#         self.drop_connect_rate = drop_connect_rate

#         # Squeeze-and-excitation
#         if self.has_se:
#             se_kwargs = resolve_se_args(se_kwargs, out_chs, act_layer)
#             se_frac = 1
#             se_mode = 'small'
#             if se_mode == 'large':
#                 self.se = SqueezeExcite(out_chs, se_ratio=se_ratio * se_frac, **se_kwargs)
#             else:
#                 self.se = SqueezeExcite(out_chs, se_ratio=se_ratio * se_frac / exp_ratio, **se_kwargs)
#         # import pdb;pdb.set_trace()

#         # # Point-wise linear projection
#         # self.conv_pwl = select_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
#         # self.bn3 = norm_layer(out_chs, **norm_kwargs)

#         self.identity = False
#         self.identity_div = 1
#         self.expand_ratio = exp_ratio

#         hidden_dim = in_chs // exp_ratio
#         # if hidden_dim < out_chs / 6.:
#         #     hidden_dim = math.ceil(out_chs / 6.)
#         #     hidden_dim = _make_divisible(hidden_dim, 16)
#         hidden_dim = int(hidden_dim)
#         # import pdb;pdb.set_trace()
#         if self.expand_ratio == 2:
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(in_chs, in_chs, dw_kernel_size, 1, dw_kernel_size//2, groups=in_chs, bias=False),
#                 nn.BatchNorm2d(in_chs),
#                 act_layer(inplace=True),
#                 # pw-linear
#                 nn.Conv2d(in_chs, hidden_dim, 1, 1, 0, bias=False, groups=group_1x1),
#                 nn.BatchNorm2d(hidden_dim),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, out_chs, 1, 1, 0, bias=False, groups=group_1x1),
#                 nn.BatchNorm2d(out_chs),
#                 act_layer(inplace=True),
#                 # dw
#                 nn.Conv2d(out_chs, out_chs, dw_kernel_size, stride, dw_kernel_size//2, groups=out_chs, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )
#         elif in_chs != out_chs and stride == 1 and keep_3x3 == False:
#             self.conv = nn.Sequential(
#                 # pw-linear
#                 nn.Conv2d(in_chs, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, out_chs, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_chs),
#                 act_layer(inplace=True),
#             )
#         elif in_chs != out_chs and stride == 2 and keep_3x3==False:
#             self.conv = nn.Sequential(
#                 # pw-linear
#                 nn.Conv2d(in_chs, hidden_dim, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 # pw-linear
#                 nn.Conv2d(hidden_dim, out_chs, 1, 1, 0, bias=False),
#                 nn.BatchNorm2d(out_chs),
#                 act_layer(inplace=True),
#                 # dw
#                 nn.Conv2d(out_chs, out_chs, dw_kernel_size, stride, dw_kernel_size//2, groups=out_chs, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )
#         else:
#             if keep_3x3 == False:
#                 self.identity = True
#             self.conv = nn.Sequential(
#                 # dw
#                 nn.Conv2d(in_chs, in_chs, dw_kernel_size, 1, dw_kernel_size//2, groups=in_chs, bias=False),
#                 nn.BatchNorm2d(in_chs),
#                 act_layer(inplace=True),
#                 # pw
#                 nn.Conv2d(in_chs, hidden_dim, 1, 1, 0, bias=False, groups = group_1x1),
#                 nn.BatchNorm2d(hidden_dim),
#                 #act_layer(inplace=True),
#                 # pw
#                 nn.Conv2d(hidden_dim, out_chs, 1, 1, 0, bias=False, groups = group_1x1),
#                 nn.BatchNorm2d(out_chs),
#                 act_layer(inplace=True),
#                 # dw
#                 nn.Conv2d(out_chs, out_chs, dw_kernel_size, 1, dw_kernel_size//2, groups=out_chs, bias=False),
#                 nn.BatchNorm2d(out_chs),
#             )
#     def feature_module(self, location):
#         if location == 'post_exp':
#             return 'act1'
#         return 'conv_pwl'

#     def feature_channels(self, location):
#         if location == 'post_exp':
#             return self.conv_pw.out_channels
#         # location == 'pre_pw'
#         return self.conv_pwl.in_channels

#     def forward(self, x):
#         residual = x
#         out = self.conv(x)

#         # Squeeze-and-excitation
#         if self.has_se:
#             x = self.se(x)

#         if self.identity:
#             if self.drop_connect_rate > 0.:
#                 out = drop_connect(out, self.training, self.drop_connect_rate)
#             out = out + x
#             return out
#         else:
#             return out
#         # # Point-wise expansion
#         # x = self.conv_pw(x)
#         # x = self.bn1(x)
#         # x = self.act1(x)

#         # # Depth-wise convolution
#         # x = self.conv_dw(x)
#         # x = self.bn2(x)
#         # x = self.act2(x)

#         # # Point-wise linear projection
#         # x = self.conv_pwl(x)
#         # x = self.bn3(x)

#         # if self.has_residual:
#         #     if self.drop_connect_rate > 0.:
#         #         x = drop_connect(x, self.training, self.drop_connect_rate)
#         #     x += residual

#         # return x
class I2RGhostBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_connect_rate=0., keep_3x3 = False, group_1x1=1):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate
        self.expand_ratio = exp_ratio

        # Get static or dynamic convolution depending on image size
        # Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        Conv2d = nn.Conv2d

        # Expansion phase
        inp = in_chs
        oup = in_chs // self.expand_ratio  # number of output channels
        final_oup = out_chs
        self.inp, self.final_oup = inp, final_oup
        self.identity = False
        
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        oup = _make_divisible(oup,2)
        k = dw_kernel_size
        s = stride

        # apply repeat scheme
        self.split_ratio = 2
        self.ghost_idx_inp = inp // self.split_ratio
        self.ghost_idx_oup = int(final_oup - self.ghost_idx_inp)

        self.inp, self.final_oup, self.s = inp, final_oup, s
        # if self._block_args.expand_ratio != 1:
        #     self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        #     self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.expand_ratio == 2:
            # self.features = nn.Sequential(
            self.dwise_conv1 = Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=k, padding=k//2, bias=False, groups=in_chs)
            self.bn1 = norm_layer(in_chs, **norm_kwargs)
            self.act = act_layer(inplace=True)
            #first linear layer
            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = norm_layer(oup, **norm_kwargs)
            # sec linear layer
            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False)
            self.bn3 = norm_layer(self.ghost_idx_oup, **norm_kwargs) 
            # act_layer(inplace=True),
            # expand layer
            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, padding=k//2,bias=False, groups = final_oup, stride=s)
            self.bn4 = norm_layer(final_oup, **norm_kwargs) 
            # )
        elif inp != final_oup and s == 1:
            # self.features=nn.Sequential(
            self.project_layer = Conv2d(in_channels=in_chs, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = norm_layer(oup, **norm_kwargs) 
            # only two linear layers are needed
            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False, groups = group_1x1)
            self.bn3 = norm_layer(final_oup, **norm_kwargs)
            self.act = act_layer(inplace=True)
            # )
        elif in_chs != final_oup and s == 2:
            # self.features = nn.Sequential(
            self.project_layer = Conv2d(in_channels=in_chs, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = norm_layer(oup, **norm_kwargs) 

            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self.bn3 = norm_layer(final_oup, **norm_kwargs)
            self.act = act_layer(inplace=True)

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k,padding=k//2, bias=False, groups = final_oup, stride=s)
            self.bn4 = norm_layer(final_oup, **norm_kwargs)
            # )
        else:
            self.identity = True
            # self.features =  nn.Sequential(
            self.dwise_conv1=Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=k, padding=k//2,bias=False, groups = in_chs)
            self.bn1 = norm_layer(in_chs, **norm_kwargs)
            self.act = act_layer(inplace=True)

            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn2 = norm_layer(oup, **norm_kwargs) 

            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn3 = norm_layer(self.ghost_idx_oup, **norm_kwargs) 
            # act_layer(inplace=True),

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, padding=k//2,bias=False, groups = final_oup)
            self.bn4 = norm_layer(final_oup, **norm_kwargs)
            # )
        if self.has_se:
            se_mode = 'large'
            if se_mode == 'large':
                se_frac = 0.5
                se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
                self.se = SqueezeExcite(out_chs, se_ratio=se_ratio * se_frac, **se_kwargs)
            else:
                se_frac = 1
                se_kwargs = resolve_se_args(se_kwargs, out_chs, act_layer)
                self.se = SqueezeExcite(out_chs, se_ratio=se_ratio * se_frac / exp_ratio, **se_kwargs)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        # x = self.features(inputs)
        if self.expand_ratio == 2:
            # first dwise conv
            x = self.act(self.bn1(self.dwise_conv1(inputs)))
            # first 1x1 conv
            ghost_id = x[:,self.ghost_idx_inp:,:,:]
            x = self.bn2(self.project_layer(x[:,:self.ghost_idx_inp,:,:]))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # generate more features
            x = torch.cat([x,ghost_id],dim=1)
            # second dwise conv
            x = self.bn4(self.dwise_conv2(x))
        elif self.inp != self.final_oup and self.s == 1:
            # first 1x1 conv
            x = self.bn2(self.project_layer(inputs))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
        elif self.inp != self.final_oup and self.s == 2:
            # first 1x1 conv
            x = self.bn2(self.project_layer(inputs))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # second dwise conv
            x = self.bn4(self.dwise_conv2(x))
        else:
            # first dwise conv
            x = self.act(self.bn1(self.dwise_conv1(inputs)))
            # first 1x1 conv
            ghost_id = x[:,self.ghost_idx_inp:,:,:]
            x = self.bn2(self.project_layer(x[:,:self.ghost_idx_inp,:,:]))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # second dwise conv
            x = torch.cat([x,ghost_id],dim=1)
            x = self.bn4(self.dwise_conv2(x))
        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Skip connection and drop connect
        # input_filters, output_filters = self.in_chs, self.out_chs
        # if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
        #     # import pdb;pdb.set_trace()
        #     if drop_connect_rate:
        #         x = drop_connect(x, p=drop_connect_rate, training=self.training)
        #     x = x + inputs  # skip connection
        # return x
        if self.identity:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x = x + inputs
            return x
        else:
            return x
class I2RBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 conv_kwargs=None, drop_connect_rate=0., keep_3x3 = False, group_1x1=2):
        super().__init__()
        norm_kwargs = norm_kwargs or {}
        conv_kwargs = conv_kwargs or {}
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate
        self.expand_ratio = exp_ratio

        # Get static or dynamic convolution depending on image size
        # Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        Conv2d = nn.Conv2d

        # Expansion phase
        inp = in_chs
        oup = in_chs // self.expand_ratio  # number of output channels
        final_oup = out_chs
        self.inp, self.final_oup = inp, final_oup
        self.identity = False
        
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        oup = _make_divisible(oup,2)
        k = dw_kernel_size
        s = stride

        # apply repeat scheme
        self.ghost_idx_inp = inp 
        self.ghost_idx_oup = final_oup

        self.inp, self.final_oup, self.s = inp, final_oup, s
        # if self._block_args.expand_ratio != 1:
        #     self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        #     self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self.expand_ratio == 2:
            # self.features = nn.Sequential(
            self.dwise_conv1 = Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=k, padding=k//2, bias=False, groups=in_chs)
            self.bn1 = norm_layer(in_chs, **norm_kwargs)
            self.act = act_layer(inplace=True)
            #first linear layer
            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False, groups = group_1x1)
            self.bn2 = norm_layer(oup, **norm_kwargs)
            # sec linear layer
            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False, groups = group_1x1)
            self.bn3 = norm_layer(self.ghost_idx_oup, **norm_kwargs) 
            # act_layer(inplace=True),
            # expand layer
            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, padding=k//2,bias=False, groups = final_oup, stride=s)
            self.bn4 = norm_layer(final_oup, **norm_kwargs) 
            # )
        elif inp != final_oup and s == 1:
            # self.features=nn.Sequential(
            self.project_layer = Conv2d(in_channels=in_chs, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = norm_layer(oup, **norm_kwargs) 
            # only two linear layers are needed
            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self.bn3 = norm_layer(final_oup, **norm_kwargs)
            self.act = act_layer(inplace=True)
            # )
        elif in_chs != final_oup and s == 2:
            # self.features = nn.Sequential(
            self.project_layer = Conv2d(in_channels=in_chs, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = norm_layer(oup, **norm_kwargs) 

            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self.bn3 = norm_layer(final_oup, **norm_kwargs)
            self.act = act_layer(inplace=True)

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k,padding=k//2, bias=False, groups = final_oup, stride=s)
            self.bn4 = norm_layer(final_oup, **norm_kwargs)
            # )
        else:
            self.identity = True
            # self.features =  nn.Sequential(
            self.dwise_conv1=Conv2d(in_channels=in_chs, out_channels=in_chs, kernel_size=k, padding=k//2,bias=False, groups = in_chs)
            self.bn1 = norm_layer(in_chs, **norm_kwargs)
            self.act = act_layer(inplace=True)

            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn2 = norm_layer(oup, **norm_kwargs) 

            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn3 = norm_layer(self.ghost_idx_oup, **norm_kwargs) 
            # act_layer(inplace=True),

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, padding=k//2,bias=False, groups = final_oup)
            self.bn4 = norm_layer(final_oup, **norm_kwargs)
            # )
        if self.has_se:
            se_mode = 'small'
            if se_mode == 'large':
                se_frac = 0.5
                se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
                self.se = SqueezeExcite(out_chs, se_ratio=se_ratio * se_frac, **se_kwargs)
            else:
                se_frac = 1
                se_kwargs = resolve_se_args(se_kwargs, out_chs, act_layer)
                self.se = SqueezeExcite(out_chs, se_ratio=se_ratio * se_frac / exp_ratio, **se_kwargs)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        # x = self.features(inputs)
        if self.expand_ratio == 2:
            # first dwise conv
            x = self.act(self.bn1(self.dwise_conv1(inputs)))
            # first 1x1 conv
            ghost_id = x[:,self.ghost_idx_inp:,:,:]
            x = self.bn2(self.project_layer(x[:,:self.ghost_idx_inp,:,:]))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # generate more features
            x = torch.cat([x,ghost_id],dim=1)
            # second dwise conv
            x = self.bn4(self.dwise_conv2(x))
        elif self.inp != self.final_oup and self.s == 1:
            # first 1x1 conv
            x = self.bn2(self.project_layer(inputs))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
        elif self.inp != self.final_oup and self.s == 2:
            # first 1x1 conv
            x = self.bn2(self.project_layer(inputs))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # second dwise conv
            x = self.bn4(self.dwise_conv2(x))
        else:
            # first dwise conv
            x = self.act(self.bn1(self.dwise_conv1(inputs)))
            # first 1x1 conv
            ghost_id = x[:,self.ghost_idx_inp:,:,:]
            x = self.bn2(self.project_layer(x[:,:self.ghost_idx_inp,:,:]))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # second dwise conv
            x = torch.cat([x,ghost_id],dim=1)
            x = self.bn4(self.dwise_conv2(x))
        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Skip connection and drop connect
        # input_filters, output_filters = self.in_chs, self.out_chs
        # if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
        #     # import pdb;pdb.set_trace()
        #     if drop_connect_rate:
        #         x = drop_connect(x, p=drop_connect_rate, training=self.training)
        #     x = x + inputs  # skip connection
        # return x
        if self.identity:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x = x + inputs
            return x
        else:
            return x
class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(self, in_chs, out_chs, dw_kernel_size=3,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False,
                 exp_ratio=1.0, exp_kernel_size=1, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 num_experts=0, drop_connect_rate=0.):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)

        super(CondConvResidual, self).__init__(
            in_chs, out_chs, dw_kernel_size=dw_kernel_size, stride=stride, dilation=dilation, pad_type=pad_type,
            act_layer=act_layer, noskip=noskip, exp_ratio=exp_ratio, exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size, se_ratio=se_ratio, se_kwargs=se_kwargs,
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, conv_kwargs=conv_kwargs,
            drop_connect_rate=drop_connect_rate)

        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        residual = x

        # CondConv routing
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))

        # Point-wise expansion
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.act1(x)

        # Depth-wise convolution
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.act2(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride"""

    def __init__(self, in_chs, out_chs, exp_kernel_size=3, exp_ratio=1.0, fake_in_chs=0,
                 stride=1, dilation=1, pad_type='', act_layer=nn.ReLU, noskip=False, pw_kernel_size=1,
                 se_ratio=0., se_kwargs=None, norm_layer=nn.BatchNorm2d, norm_kwargs=None,
                 drop_connect_rate=0.):
        super(EdgeResidual, self).__init__()
        norm_kwargs = norm_kwargs or {}
        if fake_in_chs > 0:
            mid_chs = make_divisible(fake_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        self.has_se = se_ratio is not None and se_ratio > 0.
        self.has_residual = (in_chs == out_chs and stride == 1) and not noskip
        self.drop_connect_rate = drop_connect_rate

        # Expansion convolution
        self.conv_exp = select_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type)
        self.bn1 = norm_layer(mid_chs, **norm_kwargs)
        self.act1 = act_layer(inplace=True)

        # Squeeze-and-excitation
        if self.has_se:
            se_kwargs = resolve_se_args(se_kwargs, in_chs, act_layer)
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio, **se_kwargs)

        # Point-wise linear projection
        self.conv_pwl = select_conv2d(
            mid_chs, out_chs, pw_kernel_size, stride=stride, dilation=dilation, padding=pad_type)
        self.bn2 = norm_layer(out_chs, **norm_kwargs)

    def feature_module(self, location):
        if location == 'post_exp':
            return 'act1'
        return 'conv_pwl'

    def feature_channels(self, location):
        if location == 'post_exp':
            return self.conv_exp.out_channels
        # location == 'pre_pw'
        return self.conv_pwl.in_channels

    def forward(self, x):
        residual = x

        # Expansion convolution
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.act1(x)

        # Squeeze-and-excitation
        if self.has_se:
            x = self.se(x)

        # Point-wise linear projection
        x = self.conv_pwl(x)
        x = self.bn2(x)

        if self.has_residual:
            if self.drop_connect_rate > 0.:
                x = drop_connect(x, self.training, self.drop_connect_rate)
            x += residual

        return x
