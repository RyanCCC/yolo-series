import os
from .base import Config

class YOLOV5Config(Config):
    # Train
    logdir = './yolov5/logs/'
    dataset_base_path = r'./villages'
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
    pretrain_weight = './model/yolov5_s_v6.1.h5'
    save_weight = 'village_yolov5_tf2.h5'
    anchors_path = './yolov5/data/yolov5_anchors.txt'
    phi='s'
    # Inference
    predict_weight = './model/yolov5_s_v6.1.h5'
    score=0.3
    iou=0.5
    input_shape = [640, 640]
    mosaic = True
    mosaic_prob = 0.5
    if mosaic:
        # mixup数据增强
        mixup = True
        mixup_prob = 0.5
    else:
        mixup = False
    special_aug_ratio = 0.7
    focal_loss = False
    focal_alpha = 0.25
    focal_gamma = 2
    gpus = '0'