import colorsys
import os
import sys
sys.path.append(os.getcwd())
import config as sys_config
import numpy as np
from keras import backend as K
from keras.layers import Input
from keras.models import load_model
from PIL import Image
from tqdm import tqdm

from nets.yolo4 import yolo_body, yolo_eval
from utils.utils import letterbox_image
from nets.yolo import YOLO


'''
这里设置的门限值较低是因为计算map需要用到不同门限条件下的Recall和Precision值。
所以只有保留的框足够多，计算的map才会更精确，详情可以了解map的原理。
计算map时输出的Recall和Precision值指的是门限为0.5时的Recall和Precision值。

此处获得的./input/detection-results/里面的txt的框的数量会比直接predict多一些，这是因为这里的门限低，
目的是为了计算不同门限条件下的Recall和Precision值，从而实现map的计算。

这里的self.iou指的是非极大抑制所用到的iou，具体的可以了解非极大抑制的原理，
如果低分框与高分框的iou大于这里设定的self.iou，那么该低分框将会被剔除。

如果想要设定mAP0.x，比如设定mAP0.75，可以去get_map.py设定MINOVERLAP。
'''

yolo = YOLO(    
    model_path=sys_config.model_path,
    anchors_path=sys_config.anchors_path,
    classes_path=sys_config.classes_path,
    score=sys_config.map_socre,
    iou=sys_config.map_iou,
    max_boxes=sys_config.max_boxes,
    model_image_size=(sys_config.imagesize, sys_config.imagesize),
    letterbox_image=sys_config.letterbox_image
)

image_ids = open(os.path.join(sys_config.test_txt, 'test.txt')).read().strip().split()

if not os.path.exists(os.path.join(sys_config.result, sys_config.pr_folder_name)):
    os.makedirs(os.path.join(sys_config.result, sys_config.pr_folder_name))

for image_id in tqdm(image_ids):
    image_path = sys_config.dataset_base_path+"/JPEGImages/"+image_id+".jpg"
    image = Image.open(image_path)
    # 开启后在之后计算mAP可以可视化
    # image.save("./input/images-optional/"+image_id+".jpg")
    yolo.get_dr_txt(image_id,image)
    
print("Conversion completed!")
