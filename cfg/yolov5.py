import os
from .base import Config
import datetime

class YOLOV5Config(Config):
    task = 'village_detection'
    logdir = './yolov5/logs/'

    # Dataset
    dataset_base_path = r'./VOC2007'
    classes_path = os.path.join(dataset_base_path, 'coco.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')

    # cuda setting 
    cuda = False
    gpus = '0'

    # Train
    '''
    冻结训练：Init_epoch -- Freeze_epoch
    解冻训练：Freeze_epoch -- epoch
    '''
    Freeze_Train = True
    Init_epoch = 0
    Freeze_epoch = 50
    epoch = 100
    batch_size = 16
    Unfreeze_batch_size = batch_size//2
    # learning rate
    learning_rate_freeze = 1e-3
    learning_rate_unfreeze = 1e-4
    # pretrain weight
    pretrain_weight = './yolov5v61/checkpoints/yolov5_l_v6.1.pth'
    # anchors
    anchors_path = './yolov5/data/yolov5_anchors.txt'
    phi='l'
    # save weight
    time_str = str(datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d'))
    save_weight = f'{task}_yolov5{phi}_{time_str}.h5'
    # mosaic and mix up
    mosaic = True
    mosaic_prob = 0.5
    if mosaic:
        # mixup数据增强
        mixup = True
        mixup_prob = 0.5
    else:
        mixup = False
    
    # callback
    eval_flag = True
    special_aug_ratio = 0.7

    score=0.3
    iou=0.5
    input_shape = [640, 640]

    # focal loss
    focal_loss = False
    focal_alpha = 0.25
    focal_gamma = 2

    # YOLOV5 backbone
    '''
    cspdarknet（默认）
    convnext_tiny
    convnext_small
    swin_transfomer_tiny
    '''
    backbone = 'cspdarknet'