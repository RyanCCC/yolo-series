from functools import wraps
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
from yolox.lib.utils import compose
import tensorflow_model_optimization as tfmot

class Mish(Layer, tfmot.sparsity.keras.PrunableLayer):  
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape  
    
    def get_prunable_weights(self):
        return self.weights

class SiLU(Layer):
    def __init__(self, **kwargs):
        super(SiLU, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs*K.sigmoid(inputs)
    
    def get_config(self):
        return super(SiLU, self).get_config()
    
    def compute_output_shape(self, input_shape):
        return input_shape

class Focus(Layer):
    def __init__(self):
        super(Focus, self).__init__()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2 if input_shape[1] != None else input_shape[1], input_shape[2] // 2 if input_shape[2] != None else input_shape[2], input_shape[3] * 4)
    
    def call(self,x):
        return tf.concat(
            [x[...,  ::2,  ::2, :],
             x[..., 1::2,  ::2, :],
             x[...,  ::2, 1::2, :],
             x[..., 1::2, 1::2, :]],
             axis=-1
        )


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    # darknet_conv_kwargs = {'kernel_regularizer': l2(5e-4)}
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02)}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2,2) else 'same'
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)

@wraps(Conv2D)
def DarknetConv2D_withL2(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


#DarknetConv2D + BatchNormalization + Mish
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())

def DarknetConv2D_BN_SiLU(*args, **kwargs):
    no_bias_kwargs = {'use_bias':False}
    no_bias_kwargs.update(kwargs)
    if "name" in kwargs.keys():
        no_bias_kwargs['name'] = kwargs['name'] + '.conv'
    return compose(
        DarknetConv2D_withL2(*args, **no_bias_kwargs),
        BatchNormalization(momentum = 0.97, epsilon = 0.001, name = kwargs['name'] + '.bn'),
        SiLU())



#CSPdarknet结构块
def resblock_body(x, num_filters, num_blocks, all_narrow=True):
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2))(preconv1)
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1)),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3)))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1))(mainconv)
    route = Concatenate()([postconv, shortconv])
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)

# CSPDarknet主体
def darknet_body(x):
    x = DarknetConv2D_BN_Mish(32, (3,3))(x)
    x = resblock_body(x, 64, 1, False)
    x = resblock_body(x, 128, 2)
    x = resblock_body(x, 256, 8)
    feat1 = x
    x = resblock_body(x, 512, 8)
    feat2 = x
    x = resblock_body(x, 1024, 4)
    feat3 = x
    return feat1,feat2,feat3

# 以下是YOLOx的网络结构
def SPPBottleneck(x, out_channels, weight_decay=5e-4, name=""):
    # SPP结构：不同尺度池化后进行concat
    x = DarknetConv2D_BN_SiLU(out_channels // 2, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    maxpool1 = MaxPooling2D(pool_size=(5, 5), strides=(1, 1), padding='same')(x)
    maxpool2 = MaxPooling2D(pool_size=(9, 9), strides=(1, 1), padding='same')(x)
    maxpool3 = MaxPooling2D(pool_size=(13, 13), strides=(1, 1), padding='same')(x)
    x = Concatenate()([x, maxpool1, maxpool2, maxpool3])
    x = DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    return x


def Bottleneck(x, out_channels, shortcut=True, weight_decay=5e-4, name=""):
    y = compose(
            DarknetConv2D_BN_SiLU(out_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1'),
            DarknetConv2D_BN_SiLU(out_channels, (3, 3), weight_decay=weight_decay, name = name + '.conv2'))(x)
    if shortcut:
        y = Add()([x, y])
    return y

def CSPLayer(x, num_filters, num_blocks, shortcut=True, expansion=0.5, weight_decay=5e-4, name=""):
    hidden_channels = int(num_filters * expansion)  # hidden channels
    x_1 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv1')(x)
    x_2 = DarknetConv2D_BN_SiLU(hidden_channels, (1, 1), weight_decay=weight_decay, name = name + '.conv2')(x)
    for i in range(num_blocks):
        x_1 = Bottleneck(x_1, hidden_channels, shortcut=shortcut, weight_decay=weight_decay, name = name + '.m.' + str(i))
    route = Concatenate()([x_1, x_2])
    return DarknetConv2D_BN_SiLU(num_filters, (1, 1), weight_decay=weight_decay, name = name + '.conv3')(route)

def resblock_body_yolox(x, num_filters, num_blocks, expansion=0.5, shortcut=True, last=False, weight_decay=5e-4, name = ""):
    x = ZeroPadding2D(((1, 0),(1, 0)))(x)
    x = DarknetConv2D_BN_SiLU(num_filters, (3, 3), strides = (2, 2), weight_decay=weight_decay, name = name + '.0')(x)
    if last:
        x = SPPBottleneck(x, num_filters, weight_decay=weight_decay, name = name + '.1')
    return CSPLayer(x, num_filters, num_blocks, shortcut=shortcut, expansion=expansion, weight_decay=weight_decay, name = name + '.1' if not last else name + '.2')

def darknet_body_yolox(x, dep_mul, wid_mul, weight_decay=5e-4):
    base_channels   = int(wid_mul * 64)  # 64
    base_depth      = max(round(dep_mul * 3), 1)  # 3
    # 640, 640, 3 => 320, 320, 12
    x = Focus()(x)
    # 320, 320, 12 => 320, 320, 64
    x = DarknetConv2D_BN_SiLU(base_channels, (3, 3), weight_decay=weight_decay, name = 'backbone.backbone.stem.conv')(x)
    # 320, 320, 64 => 160, 160, 128
    x = resblock_body_yolox(x, base_channels * 2, base_depth, weight_decay=weight_decay, name = 'backbone.backbone.dark2')
    # 160, 160, 128 => 80, 80, 256
    x = resblock_body_yolox(x, base_channels * 4, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark3')
    feat1 = x
    # 80, 80, 256 => 40, 40, 512
    x = resblock_body_yolox(x, base_channels * 8, base_depth * 3, weight_decay=weight_decay, name = 'backbone.backbone.dark4')
    feat2 = x
    # 40, 40, 512 => 20, 20, 1024
    x = resblock_body_yolox(x, base_channels * 16, base_depth, shortcut=False, last=True, weight_decay=weight_decay, name = 'backbone.backbone.dark5')
    feat3 = x
    return feat1,feat2,feat3