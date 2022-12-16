
from tools.tfrecord_create import main_create_tfrecord
import numpy as np
from yolov4.lib.dataloader import preprocess_true_boxes
tfrecord_save_path= './validate.tfrecord'
dataset_root = './VOC2007'
class_path = './VOC2007/voc.names'
dataset_type='val'

# 测试preprocess_true_boxes

'''
debug preprocess_true_boxes
'''
true_boxes = [[[263, 211, 324, 339, 8], [165, 264, 253, 372, 8], [241, 194, 295, 299, 8], [150, 141, 229, 284, 14]],
              [[69, 172, 270, 330, 12], [150, 141, 229, 284, 14], [241, 194, 295, 299, 8], [285, 201, 327, 331, 14]]]

true_boxes = np.array(true_boxes)
print(true_boxes.shape)
input_shape = (416, 416)
anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
num_classes = 20

preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes)


main_create_tfrecord(tfrecord_save_path=tfrecord_save_path, dataset_root=dataset_root, class_path=class_path, dataset_type=dataset_type)
print('finish')