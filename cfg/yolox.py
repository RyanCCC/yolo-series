from .base import Config
import os
import datetime

class YOLOXConfig(Config):
    task = 'village_detection'
    # dataset
    logdir = './yolox/logs/'
    dataset_base_path = r'./VOC2007'
    classes_path = os.path.join(dataset_base_path, 'voc.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    
    time_str = str(datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d'))
    phi= 's'
    save_weight = f'{task}_yolox_{phi}_{time_str}.h5'
    score=0.5
    iou=0.5
    pretrain_weight = './yolox/checkpoints/yolox_s.pth'
    
    # cuda
    cuda = False
    gpus = '0'
    # train
    learning_rate = 1e-2
    Freeze_Train = True
    Freeze_epoch = 50
    epoch = 100
    batch_size = 32

    input_shape = [640, 640]

    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio   = 0.7

    eval_flag = True

