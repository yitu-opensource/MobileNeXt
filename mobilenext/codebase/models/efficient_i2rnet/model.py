import torch
from torch import nn
from torch.nn import functional as F
from .activations import sigmoid, HardSwish, Swish
from .utils_i2rnet import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    Conv2dDynamicSamePadding,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
)

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

class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d
        padding = self._block_args.kernel_size //2
        # Conv2d = Conv2dDynamicSamePadding

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False, padding = padding)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

class I2RConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d
        padding = self._block_args.kernel_size //2

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters // self._block_args.expand_ratio  # number of output channels
        final_oup = self._block_args.output_filters
        self.inp, self.final_oup = inp, final_oup
        self.identity = False
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        k = self._block_args.kernel_size
        s = self._block_args.stride[0] if isinstance(self._block_args.stride,list) else self._block_args.stride
        if self._block_args.expand_ratio == 2:
            self._project_conv = Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups=inp)
            self._bn0 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
 
            self._linear1 = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
             
            self._linear2 = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
             
            self._expand_conv = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, 
                                      stride = s, groups = final_oup)
            self._bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        elif inp != final_oup and s == 1:
            self._project_conv = None
            self._expand_conv = None
            self._linear1 = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self._linear2 = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        elif inp != final_oup and s == 2:
            self._project_conv = None
            self._linear1 = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self._linear2 = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

            self._expand_conv = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, 
                                     stride = s, groups = final_oup)
            self._bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        else:
            # if inp == final_oup:
            self._project_conv = Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups = inp)
            self._bn0 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self._expand_conv = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup)
            self._bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # if not (self._block_args.expand_ratio == 2):
            self.identity = True

            self._linear1 = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            self._linear2 = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps) # Depthwise convolution phase
        # self._depthwise_conv = Conv2d(
        #     in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(final_oup / self._block_args.expand_ratio  * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=final_oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=final_oup, kernel_size=1)

        # # Output phase
        # self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        x = inputs
        # NOTE:remove the first 3x3 conv to reduce running mem, need to verfy the performance
        if self._project_conv is not None:
            x = relu_fn(self._bn0(self._project_conv(inputs)))
        x = self._bn1(self._linear1(x))
        x = relu_fn(self._bn2(self._linear2(x)))
        if self._expand_conv is not None:
            x = self._bn3(self._expand_conv(x))
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class MBConvBlockV1(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params, mgroup=1):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters // self._block_args.expand_ratio  # number of output channels
        final_oup = self._block_args.output_filters
        self.inp, self.final_oup = inp, final_oup
        group_1x1 = mgroup
        self.identity = False
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        oup = _make_divisible(oup,2)
        k = self._block_args.kernel_size
        s = self._block_args.stride[0] if isinstance(self._block_args.stride,list) else self._block_args.stride
        # if self._block_args.expand_ratio != 1:
        #     self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        #     self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self._block_args.expand_ratio == 2:
            self.features = nn.Sequential(
            Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups=inp),
            nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps),
            Swish(),
            #first linear layer
            Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1),
            nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps),
            # sec linear layer
            Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False, groups=group_1x1),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            Swish(),
            # expand layer
            Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            )
        elif inp != final_oup and s == 1:
            self.features=nn.Sequential(
            Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps),
            # only two linear layers are needed
            Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            Swish(),
            )
        elif inp != final_oup and s == 2:
            self.features = nn.Sequential(
            Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps),

            Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            Swish(),

            Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            )
        else:
            self.identity = True
            self.features =  nn.Sequential(
            Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups = inp),
            nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps),
            Swish(),

            Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1),
            nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps),

            Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False, groups=group_1x1),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            Swish(),

            Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup),
            nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps),
            )
        # Depthwise convolution phase
        # self._depthwise_conv = Conv2d(
        #     in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        # import pdb;pdb.set_trace()
        if self.has_se:
            se_expand_ratio = 1
            # num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio * se_expand_ratio))
            num_squeezed_channels = max(1, int(final_oup / self._block_args.expand_ratio * self._block_args.se_ratio * se_expand_ratio))
            self._se_reduce = Conv2d(in_channels=final_oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=final_oup, kernel_size=1)

        # # Output phase
        # self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        x = self.features(inputs)
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
class GhostI2RBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        group_1x1 = 1

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters // self._block_args.expand_ratio  # number of output channels
        final_oup = self._block_args.output_filters
        self.inp, self.final_oup = inp, final_oup
        self.identity = False
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        oup = _make_divisible(oup,2)
        k = self._block_args.kernel_size
        s = self._block_args.stride[0] if isinstance(self._block_args.stride,list) else self._block_args.stride

        # apply repeat scheme
        self.split_ratio = 2
        self.ghost_idx_inp = inp // self.split_ratio
        self.ghost_idx_oup = int(final_oup - self.ghost_idx_inp)

        self.inp, self.final_oup, self.s = inp, final_oup, s
        # if self._block_args.expand_ratio != 1:
        #     self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        #     self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self._block_args.expand_ratio == 2:
            # self.features = nn.Sequential(
            self.dwise_conv1 = Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups=inp)
            self.bn1 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()
            #first linear layer
            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # sec linear layer
            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=self.ghost_idx_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # Swish(),
            # expand layer
            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        elif inp != final_oup and s == 1:
            # self.features=nn.Sequential(
            self.project_layer = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # only two linear layers are needed
            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False, groups = group_1x1)
            self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()
            # )
        elif inp != final_oup and s == 2:
            # self.features = nn.Sequential(
            self.project_layer = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        else:
            self.identity = True
            # self.features =  nn.Sequential(
            self.dwise_conv1=Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups = inp)
            self.bn1 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()

            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn3 = nn.BatchNorm2d(num_features=self.ghost_idx_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # Swish(),

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        # Depthwise convolution phase
        # self._depthwise_conv = Conv2d(
        #     in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        # import pdb;pdb.set_trace()
        if self.has_se:
            se_mode = 'large'
            if se_mode == 'large':
                se_expand_ratio = 0.5
                num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio * se_expand_ratio))
            else:
                se_expand_ratio = 1
                num_squeezed_channels = max(1, int(final_oup / self._block_args.expand_ratio * self._block_args.se_ratio * se_expand_ratio))
            self._se_reduce = Conv2d(in_channels=final_oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=final_oup, kernel_size=1)

        # # Output phase
        # self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        # x = self.features(inputs)
        if self._block_args.expand_ratio == 2:
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
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
            # import pdb;pdb.set_trace()
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
class GhostI2RBlock_change_droppath_pos(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        group_1x1 = 1
        apply_ghost = True

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters // self._block_args.expand_ratio  # number of output channels
        final_oup = self._block_args.output_filters
        self.inp, self.final_oup = inp, final_oup
        self.identity = False
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        oup = _make_divisible(oup,2)
        k = self._block_args.kernel_size
        s = self._block_args.stride[0] if isinstance(self._block_args.stride,list) else self._block_args.stride
        if apply_ghost:
        # apply repeat scheme
            self.split_ratio = 2
            self.ghost_idx_inp = inp // self.split_ratio
            self.ghost_idx_oup = int(final_oup - self.ghost_idx_inp)
        else:
            self.ghost_idx_inp = inp
            self.ghost_idx_oup = final_oup

        self.inp, self.final_oup, self.s = inp, final_oup, s
        # if self._block_args.expand_ratio != 1:
        #     self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        #     self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self._block_args.expand_ratio == 2:
            # self.features = nn.Sequential(
            self.dwise_conv1 = Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups=inp)
            self.bn1 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()
            #first linear layer
            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # sec linear layer
            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=self.ghost_idx_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # Swish(),
            # expand layer
            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        elif inp != final_oup and s == 1:
            # self.features=nn.Sequential(
            self.project_layer = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # only two linear layers are needed
            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False, groups = group_1x1)
            self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()
            # )
        elif inp != final_oup and s == 2:
            # self.features = nn.Sequential(
            self.project_layer = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        else:
            self.identity = True
            # self.features =  nn.Sequential(
            self.dwise_conv1=Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups = inp)
            self.bn1 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()

            self.project_layer = Conv2d(in_channels=self.ghost_idx_inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.ghost_idx_oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn3 = nn.BatchNorm2d(num_features=self.ghost_idx_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # Swish(),

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        # Depthwise convolution phase
        # self._depthwise_conv = Conv2d(
        #     in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        # import pdb;pdb.set_trace()
        if self.has_se:
            se_expand_ratio = 0.5
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio * se_expand_ratio))
            # num_squeezed_channels = max(1, int(final_oup / self._block_args.expand_ratio * self._block_args.se_ratio * se_expand_ratio))
            self._se_reduce = Conv2d(in_channels=final_oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=final_oup, kernel_size=1)

        # # Output phase
        # self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        # x = self.features(inputs)
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self._block_args.expand_ratio == 2:
            # first dwise conv
            x = self.act(self.bn1(self.dwise_conv1(inputs)))
            # first 1x1 conv
            ghost_id = x[:,self.ghost_idx_inp:,:,:]
            x = self.bn2(self.project_layer(x[:,:self.ghost_idx_inp,:,:]))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # generate more features
            x = torch.cat([x,ghost_id],dim=1)
            if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
                if drop_connect_rate:
                    x = drop_connect(x, p=drop_connect_rate, training=self.training)
            # second dwise conv
            x = self.bn4(self.dwise_conv2(x))
        elif self.inp != self.final_oup and self.s == 1:
            # first 1x1 conv
            x = self.bn2(self.project_layer(inputs))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
                if drop_connect_rate:
                    x = drop_connect(x, p=drop_connect_rate, training=self.training)
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
            if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
                if drop_connect_rate:
                    x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = self.bn4(self.dwise_conv2(x))
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
            # if drop_connect_rate:
            #     x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
class NESI2RBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect
        group_1x1 = 1

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters // self._block_args.expand_ratio  # number of output channels
        final_oup = self._block_args.output_filters
        self.inp, self.final_oup = inp, final_oup
        self.identity = False
        if oup < oup / 6.:
           oup = math.ceil(oup / 6.)
           oup = _make_divisible(oup,16)
        oup = _make_divisible(oup,2)
        k = self._block_args.kernel_size
        s = self._block_args.stride[0] if isinstance(self._block_args.stride,list) else self._block_args.stride

        # apply repeat scheme
        self.split_ratio = 2
        self.nes_idx_inp = inp // self.split_ratio
        self.nes_idx_oup = final_oup // self.split_ratio

        self.inp, self.final_oup, self.s = inp, final_oup, s
        # if self._block_args.expand_ratio != 1:
        #     self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
        #     self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        if self._block_args.expand_ratio == 2:
            # self.features = nn.Sequential(
            self.dwise_conv1 = Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups=inp)
            self.bn1 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()
            #first linear layer
            self.project_layer = Conv2d(in_channels=self.nes_idx_inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # sec linear layer
            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.nes_idx_oup, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=self.nes_idx_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # Swish(),
            # expand layer
            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        elif inp != final_oup and s == 1:
            # self.features=nn.Sequential(
            self.project_layer = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # only two linear layers are needed
            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False, groups = group_1x1)
            self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()
            # )
        elif inp != final_oup and s == 2:
            # self.features = nn.Sequential(
            self.project_layer = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self.expand_layer = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
            self.bn3 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup, stride=s)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        else:
            self.identity = True
            # self.features =  nn.Sequential(
            self.dwise_conv1=Conv2d(in_channels=inp, out_channels=inp, kernel_size=k, bias=False, groups = inp)
            self.bn1 = nn.BatchNorm2d(num_features=inp, momentum=self._bn_mom, eps=self._bn_eps)
            self.act = Swish()

            self.project_layer = Conv2d(in_channels=self.nes_idx_inp, out_channels=oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn2 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

            self.expand_layer = Conv2d(in_channels=oup, out_channels=self.nes_idx_oup, kernel_size=1, bias=False, groups=group_1x1)
            self.bn3 = nn.BatchNorm2d(num_features=self.nes_idx_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # Swish(),

            self.dwise_conv2 = Conv2d(in_channels=final_oup, out_channels=final_oup, kernel_size=k, bias=False, groups = final_oup)
            self.bn4 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
            # )
        # Depthwise convolution phase
        # self._depthwise_conv = Conv2d(
        #     in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
        #     kernel_size=k, stride=s, bias=False)
        # self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        # import pdb;pdb.set_trace()
        if self.has_se:
            se_expand_ratio = 0.5
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio * se_expand_ratio))
            # num_squeezed_channels = max(1, int(final_oup / self._block_args.expand_ratio * self._block_args.se_ratio * se_expand_ratio))
            self._se_reduce = Conv2d(in_channels=final_oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=final_oup, kernel_size=1)

        # # Output phase
        # self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        # self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        # import pdb;pdb.set_trace()
        # x = self.features(inputs)
        if self._block_args.expand_ratio == 2:
            # first dwise conv
            x = self.act(self.bn1(self.dwise_conv1(inputs)))
            # first 1x1 conv
            nes_x = x[:,:self.nes_idx_inp,:,:] + x[:,self.nes_idx_inp:,:,:]
            x = self.bn2(self.project_layer(nes_x))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # generate more features
            x = torch.cat([x,x],dim=1)
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
            nes_x = x[:,:self.nes_idx_inp,:,:] + x[:,self.nes_idx_inp:,:,:]
            x = self.bn2(self.project_layer(nes_x))
            # second 1x1 conv
            x = self.act(self.bn3(self.expand_layer(x)))
            # second dwise conv
            x = torch.cat([x,x],dim=1)
            x = self.bn4(self.dwise_conv2(x))
        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.identity and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, mgroup=1):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)
        # Conv2d = nn.Conv2d

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        self.mgroup=mgroup
        # NOTE change first filter to be 16 to follow MOBILENETV3
        # NOTE change back to 32 for efficientnet series
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        # build_block = NESI2RBlock
        # build_block = GhostI2RBlock
        # build_block = GhostI2RBlock_change_droppath_pos
        build_block = MBConvBlockV1
        # build_block = I2RConvBlock
        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(build_block(block_args, self._global_params, mgroup=mgroup))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(build_block(block_args, self._global_params, mgroup=mgroup))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        # self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        # with torch.autograd.profiler.profile(use_cuda=True) as profile:
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
        #print(profile)
        #import pdb;pdb.set_trace()

        # Head
        # x = relu_fn(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None, mgroup=1):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return EfficientNet(blocks_args, global_params, mgroup)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000):
        model = EfficientNet.from_name(model_name, override_params={'num_classes': num_classes})
        load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet_b'+str(i) for i in range(num_models)] + ['i2rnet_b' + str(i) for i in range(num_models)]
        if model_name.replace('-','_') not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
def efficient_i2rnet(progress=None,width_mult=1, rm_1x1=None, interpolation=None, group_1x1=None):
    return EfficientNet.from_name('efficientnet-b0')

