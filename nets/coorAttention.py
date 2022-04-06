'''
Coordinate Attention
paper: Coordinate Attention for Efficient Mobile Network Design
'''
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Lambda, BatchNormalization, AveragePooling2D, ReLU, concatenate, Add
import tensorflow.keras.backend as K

def CoorAtt(inputs, reduction=32, train=False):
    def coor_att(x):
        tmpx = (ReLU(max_value=6)(x + 3)) / 6
        x = x * tmpx
        return x
    x_shape = inputs.shape.as_list()
    [b, h, w, c] = x_shape
    x_h = AveragePooling2D(pool_size=(1, w), strides=(1, 1), data_format='channels_last')(inputs)
    x_w = AveragePooling2D(pool_size=(h, 1), strides=(1, 1), data_format='channels_last')(inputs)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    y = concatenate(inputs=[x_h, x_w], axis=1)
    mip = max(8, c // reduction)
    y = Conv2D(filters=mip, kernel_size=(1, 1), strides=(1, 1), padding='valid')(y)
    y = BatchNormalization(trainable=train)(y)
    y = coor_att(y)
    x_h, x_w = Lambda(tf.split, arguments={'axis': 1, 'num_or_size_splits': [h, w]})(y)
    x_w = K.permute_dimensions(x_w, [0, 2, 1, 3])
    a_h = Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="sigmoid")(x_h)
    a_w = Conv2D(filters=c, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation="sigmoid")(x_w)
    out = inputs * a_h * a_w
    return out


if __name__ == '__main__':
    feature_input = K.ones((100, 120, 120, 64))
    attention = CoorAtt(feature_input)
    output = Add()([attention, feature_input])
    print(K.shape(output))
