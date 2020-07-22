"""
Creates a MobileNeXt Model as defined in:
Zhou Daquan, Qibin Hou, Yunpeng Chen, Jiashi Feng, Shuicheng Yan
Rethinking Bottleneck Structure for Efficient Mobile Network Design
arXiv preprint arXiv:2007.02269.
import from https://github.com/d-li14/mobilenetv2.pytorch
"""

import torch.nn as nn
import math

__all__ = ['mobilenext']


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


class SandGlass(nn.Module):
    def __init__(self, inp, oup, stride, reduction_ratio):
        super(SandGlass, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp // reduction_ratio)
        self.identity = stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # dw
            nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            # pw
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU6(inplace=True),
            # dw-linear
            nn.Conv2d(oup, oup, 3, stride, 1, groups=oup, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNeXt(nn.Module):
    def __init__(self, num_classes=1000, width_mult=1.):
        super(MobileNeXt, self).__init__()
        # setting of sandglass blocks
        self.cfgs = [
            # t, c, n, s
            [2,   96, 1, 2],
            [6,  144, 1, 1],
            [6,  192, 3, 2],
            [6,  288, 3, 2],
            [6,  384, 4, 1],
            [6,  576, 4, 2],
            [6,  960, 3, 1],
            [6, 1280, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = SandGlass
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            for i in range(n):
                layers.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
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

def mobilenext(**kwargs):
    """
    Constructs a MobileNeXt model
    """
    return MobileNeXt(**kwargs)

