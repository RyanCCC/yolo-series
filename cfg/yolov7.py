from .base import Config
import os

class YOLOV7Config(Config):
    task = "village_Detection"
    logdir = './yolov7/logs/'
    dataset_base_path = r'./villages'
    classes_path = os.path.join(dataset_base_path, 'village.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    Init_epoch = 0
    Freeze_epoch = 50
    epoch = 100
    batch_size = 16
    anchor_path = './yolov7/data/yolo_anchors.txt'
    # x or l
    phi= 'x'
    score=0.3
    iou=0.5
    input_shape = [640, 640]
    gpus = '1'
    pretrain_weight = './yolov7/checkpoints/yolov7_x_weights.h5'
    save_weight = 'village_yolov7.h5'
    # 0:DEBUG；1：INFO；2：warning；3：error
    log_level = '1'


