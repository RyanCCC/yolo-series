from .base import Config
import os
import datetime

class YOLOV7Config(Config):
    task = "village_Detection"

    # dataset
    dataset_base_path = r'./VOC2007'
    classes_path = os.path.join(dataset_base_path, 'coco.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    
    # freeze train
    Freeze_Train = True
    learning_rate = 1e-2
    Init_epoch = 0
    Freeze_epoch = 50
    epoch = 100
    batch_size = 16
    Unfreeze_batch_size = batch_size//2


    anchors_path = './yolov7/data/yolo_anchors.txt'
    score=0.3
    iou=0.5
    input_shape = [640, 640]

    # mosaic and mix up
    mosaic = True
    mosaic_prob = 0.5
    if mosaic:
        # mixup数据增强
        mixup = True
        mixup_prob = 0.5
    else:
        mixup = False
    special_aug_ratio = 0.7
    
    # GPU SETTING 
    cuda = False
    gpus = '1'
    
    # callback
    early_stopping = False
    eval_flag = True
    # 分布式训练
    distributed = False
    # DDP模式多卡可用
    sync_bn = False
    fp16 = False
    label_smoothing = 0
    # tiny: './yolov7/checkpoints/yolov7_tiny_weights.pth'

    # weight init
    # x or l
    phi= 'l'
    
    time = str(datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d'))
    tiny =False
    if tiny:
        pretrain_weight = './yolov7/checkpoints/yolov7_weights.pth'
        save_weight = f'{task}_yolov7_tiny_{time}.pth'
        best_weight = f'{task}_yolov7_tiny_{time}_best.pth'
        logdir = f'./yolov7/logs/yolov7tiny_{time}'
    else:
        pretrain_weight = './yolov7/checkpoints/yolov7_tiny_weights.pth'
        save_weight = f'{task}_yolov7_{phi}_{time}.pth'
        best_weight = f'{task}_yolov7_{phi}_{time}_best.pth'
        logdir = f'./yolov7/logs/yolov7{phi}_{time}'
    # 0:DEBUG；1：INFO；2：warning；3：error
    log_level = '1'
    
    


