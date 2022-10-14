import os
from .base import Config

class YOLOV4Config(Config):
    # Train
    logdir = './yolov4/logs/'
    dataset_base_path = r'.\villages'
    classes_path = os.path.join(dataset_base_path, 'village.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    '''
    冻结训练：Init_epoch -- Freeze_epoch
    解冻训练：Freeze_epoch -- epoch
    '''
    Init_epoch = 0
    Freeze_epoch = 50
    epoch = 100
    batch_size = 16
    learning_rate_freeze = 1e-3
    learning_rate_unfreeze = 1e-4
    pretrain_weight = './model/yolo4tf2_weight.h5'
    save_weight = 'village_tf2.h5'
    anchors_path = './yolov4/data/yolo_anchors.txt'
    # Inference
    predict_weight = './model/yolo4_voc_weights.h5'
    ISTINY=False
    ATTENTION=0
    score=0.3
    iou=0.5
    imagesize=512

