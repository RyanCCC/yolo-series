from .base import Config
import os
import datetime

class YOLOXConfig(Config):
    task = 'village_detection'
    gpus = '0'
    logdir = './yolox/logs/'
    learning_rate = 1e-2
    dataset_base_path = r'./villages'
    classes_path = os.path.join(dataset_base_path, 'village.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    Freeze_epoch = 50
    epoch = 100
    batch_size = 32
    time_str = str(datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d'))
    phi= 's'
    save_weight = f'{task}_yolox_{phi}_{time_str}.h5'
    score=0.5
    iou=0.5
    pretrain_weight = './model/village_yolox_0510.h5'
    input_shape = [640, 640]

