# from utils.tfrecord_create import main_create_tfrecord
# import numpy as np
# from utils.dataloader import preprocess_true_boxes
# tfrecord_save_path= './validate.tfrecord'
# dataset_root = './villages'
# class_path = './villages/village.names'
# dataset_type='val'

# # 测试preprocess_true_boxes

# '''
# debug preprocess_true_boxes
# '''
# true_boxes = [[[263, 211, 324, 339, 8], [165, 264, 253, 372, 8], [241, 194, 295, 299, 8], [150, 141, 229, 284, 14]],
#               [[69, 172, 270, 330, 12], [150, 141, 229, 284, 14], [241, 194, 295, 299, 8], [285, 201, 327, 331, 14]]]

# true_boxes = np.array(true_boxes)
# print(true_boxes.shape)
# input_shape = (416, 416)
# anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
# num_classes = 20
    
# preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes)


# main_create_tfrecord(tfrecord_save_path=tfrecord_save_path, dataset_root=dataset_root, class_path=class_path, dataset_type=dataset_type)
# print('finish')

import tensorflow as tf
import numpy as np
import pandas as pd
import os

from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.layers import Flatten, Dense, BatchNormalization, Activation, Dropout, Lambda, Input, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img

base_path = 'D:\Code\Exercise\dog-breed'

# 读取标签
label_path = os.path.join(base_path, 'labels.csv')
labels = pd.read_csv(label_path)
print(labels.head())
print(labels.describe())
# 查看数据的格式
#Create list of alphabetically sorted labels.
classes = sorted(list(set(labels['breed'])))
n_classes = len(classes)
print('Total unique breed {}'.format(n_classes))
#Map each label string to an integer label.
class_to_num = dict(zip(classes, range(n_classes)))
img_size = (331, 331, 3)

def images_to_array(directory, label_dataframe, target_size = img_size):
    image_labels = label_dataframe['breed']
    images = label_dataframe['id']
    # images = np.zeros([len(label_dataframe), target_size[0], target_size[1], target_size[2]],dtype=np.uint8)
    y = np.zeros([len(label_dataframe),1],dtype = np.uint8)
    for ix, image_name in enumerate(label_dataframe['id'].values):
        # img_dir = os.path.join(directory, image_name+'.jpg')
        # img = load_img(img_dir, target_size=target_size)
        # images[ix]=img
        # del img

        dog_breed = image_labels[ix]
        y[ix] = class_to_num[dog_breed]
    y = to_categorical(y)
    filenames = [os.path.join(directory, img+'.jpg') for img in label_dataframe['id'] ]
    # return images, y
    return filenames, y

image_path = os.path.join(base_path, 'train')
X, target_labels_encoded = images_to_array(image_path, labels[:])
print('Debug')