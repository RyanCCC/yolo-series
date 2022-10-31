from .base import Config
import os

class YOLOXConfig(Config):
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
    save_weight = 'village_yolox.h5'
    phi= 's'
    score=0.5
    iou=0.5
    input_shape = [640, 640]

