import os

class Config(object):
    eager=False
    # 是否对损失进行归一化，用于改变loss的大小
    normalize = False
    # 马赛克数据增强
    mosaic = False
    # 余弦退火学习率
    Cosine_scheduler = False
    # 标签平滑，0.01以下一般 如0.01、0.005
    label_smoothing = 0
    regularization = True
    '''
    冻结训练：Init_epoch -- Freeze_epoch
    解冻训练：Freeze_epoch -- epoch
    '''
    Init_epoch = 0
    Freeze_epoch = 50
    epoch = 100
    freeze_layers = 249
    freeze_layers_tiny = 60
    batch_size = 16
    learning_rate_freeze = 1e-3
    learning_rate_unfreeze = 1e-4
    # predict
    ISTINY=False
    score=0.3
    iou=0.5
    max_boxes=100
    letterbox_image=False
    ATTENTION=0
    ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]

class YOLOV4Config(Config):
    # Train
    logdir = './logs/'
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
    score=0.3
    iou=0.5
    imagesize=512


class YOLOV5Config(Config):
    # Train
    logdir = './logs/'
    dataset_base_path = r'.\villages'
    # classes_path = os.path.join(dataset_base_path, 'village.names') 
    classes_path = './data/voc.names'
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
    anchors_path = './data/yolo_anchors.txt'
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



class YOLOXConfig(Config):
    gpus = '0'
    logdir = './logs/'
    learning_rate = 1e-2
    dataset_base_path = r'.\villages'
    classes_path = os.path.join(dataset_base_path, 'village.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    Freeze_epoch = 50
    epoch = 100
    batch_size = 32
    pretrain_weight = './model/yolo4tf2_weight.h5'
    save_weight = 'village_yolox.h5'
    phi= 's'
    score=0.5
    iou=0.5
    input_shape = [640, 640]


class YOLOV7Config(Config):
    task = "village_Detection"
    logdir = './logs/'
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



class YOLOPConfig(Config):
    pass