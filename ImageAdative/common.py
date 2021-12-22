import tensorflow as tf
from tensorflow.keras.layers import *


# 定义卷积操作
def convolutional(input_data, channel, downsample=False, activate=True, bn=True):
    if downsample:
        strides = 2
        padding = 'valid'
    else:
        strides = 1
        padding= 'same'
    
    x = Conv2D(channel, 3, strides=strides, padding=padding, kernel_initializer=tf.keras.initializers.random_normal(stddev=0.02))(input_data)
    if bn:
        x = BatchNormalization()(x)
    if activate == True:
        x = Activation('relu')(x)
    return x

def extract_parameters(input_data, output_dim):
    channel = 16
    x = convolutional(input_data, channel, True, True, False)
    x = convolutional(x, 2*channel, True, True, False)
    x = convolutional(x, 2*channel, True, True, False)
    x = convolutional(x, 2*channel, True, True, False)
    x = convolutional(x, 2*channel, True, True, False)
    # 全连接层
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(output_dim, kernel_initializer=tf.keras.initializers.he_normal())(x)
    return x




    

