import tensorflow as tf
from tensorflow.keras.layers import Concatenate, ZeroPadding2D, MaxPooling2D
from .CSPdarknet53 import DarknetConv2D_BN_SiLU, Bottleneck

def SPPBottleneck(x, out_channels, weight_decay=5e-4, name = ""):
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool1)
    maxpool3 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(maxpool2)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    return x

def C3(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv1')(x)
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.cv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    route = Concatenate()([x_1, x_2])
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.cv3')(route)

def resblock_body(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):

    # 320, 320, 64 => 160, 160, 128
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    x = C3(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1')
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.2')
    return x


def darknet_body(x, base_channels, base_depth, weight_decay=5e-4):
    # 640, 640, 3 => 320, 320, 64
    x = ZeroPadding2D(((2, 2),(2, 2)))(x)
    x = DarknetConv2D_BN_SiLU(base_channels, (6, 6), strides = (2, 2), weight_decay=weight_decay, name = 'backbone.stem.conv')(x)
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.dark2')
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body(x, base_channels * 4, base_depth * 2, weight_decay=weight_decay, name = 'backbone.dark3')
    feat1 = x
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body(x, base_channels * 16, base_depth, last=True, weight_decay=weight_decay, name = 'backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3