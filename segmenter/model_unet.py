# U-Net Model
# Note: I modify the U-Net to get the ouput of the same shape as input

from mxnet.gluon import nn, loss as gloss, data as gdata
from mxnet import autograd, nd, init, image
import numpy as np
# import logging

# logging.basicConfig(level=logging.CRITICAL)

class BaseConvBlock(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(BaseConvBlock, self).__init__(**kwargs)
        # no-padding in the paper
        # here, I use padding to get the output of the same shape as input

        self.conv1 = nn.Conv2D(channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm()
        self.conv2 = nn.Conv2D(channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm()

    def hybrid_forward(self, F, x, *args, **kwargs):
        # BatchNorm input will typically be unnormalized activations from the previous layer,
        # and the output will be the normalized activations ready for the next layer.

        x = F.relu(self.conv1(x))
        # logging.info(x.shape)
        return F.relu(self.conv2(x))    
    
    def hybrid_forward__(self, F, x, *args, **kwargs):
        x = F.relu(self.conv1(x))
        # logging.info(x.shape)
        return F.relu(self.conv2(x))


class DownSampleBlock(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(DownSampleBlock, self).__init__(**kwargs)
        self.maxPool = nn.MaxPool2D(pool_size=2, strides=2)
        self.conv = BaseConvBlock(channels)

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.maxPool(x)
        # logging.info(x.shape)
        return self.conv(x)


class UpSampleBlock(nn.HybridSequential):
    def __init__(self, channels, **kwargs):
        super(UpSampleBlock, self).__init__(**kwargs)
        self.channels = channels
        self.up = nn.Conv2DTranspose(channels, kernel_size=4, padding=1, strides=2)
        self.conv = BaseConvBlock(channels)

    def hybrid_forward(self, F, x1, *args, **kwargs):
        x2 = args[0]
        x1 = self.up(x1)
        # The same as paper
        # x2 = x2[:, :, :x1.shape[2], : x1.shape[3]]

        # Fill in x1 shape to be the same as the x2
        diffY = x2.shape[2] - x1.shape[2]
        diffX = x2.shape[3] - x1.shape[3]
        x1 = nd.pad(x1,
                    mode='constant',
                    constant_value=0,
                    pad_width=(0, 0, 0, 0,
                               diffY // 2, diffY - diffY // 2,
                               diffX // 2, diffX - diffX // 2))
        x = nd.concat(x1, x2, dim=1)
        # logging.info(x.shape)
        return self.conv(x)


class UNet(nn.HybridSequential):
    def __init__(self, channels, num_class, **kwargs):
        super(UNet, self).__init__(**kwargs)

        # contracting path -> encoder
        self.input_conv = BaseConvBlock(64)
        for i in range(4):
            setattr(self, 'down_conv_%d' % i, DownSampleBlock(channels * 2 ** (i + 1)))
        # expanding path  -> decoder
        for i in range(4):
            setattr(self, 'up_conv_%d' % i, UpSampleBlock(channels * 16 // (2 ** (i + 1))))
        self.output_conv = nn.Conv2D(num_class, kernel_size=1)

    def hybrid_forward(self, F, x, *args, **kwargs):
        # logging.info('Contracting Path:')
        x1 = self.input_conv(x)
        # logging.info(x1.shape)
        x2 = getattr(self, 'down_conv_0')(x1)
        # logging.info(x2.shape)
        x3 = getattr(self, 'down_conv_1')(x2)
        # logging.info(x3.shape)
        x4 = getattr(self, 'down_conv_2')(x3)
        # logging.info(x4.shape)
        x5 = getattr(self, 'down_conv_3')(x4)
        # logging.info(x5.shape)
        # logging.info('Expansive Path:')
        x = getattr(self, 'up_conv_0')(x5, x4)
        # logging.info(x.shape)
        x = getattr(self, 'up_conv_1')(x, x3)
        # logging.info(x.shape)
        x = getattr(self, 'up_conv_2')(x, x2)
        # logging.info(x.shape)
        x = getattr(self, 'up_conv_3')(x, x1)
        # logging.info(x.shape)
        return self.output_conv(x)
