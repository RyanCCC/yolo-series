import os
from .base import Config

class YOLOV4Config(Config):
    # dataset
    logdir = './yolov4/logs/'
    dataset_base_path = r'./VOC2007'
    classes_path = os.path.join(dataset_base_path, 'voc.names') 
    train_txt= os.path.join(dataset_base_path, 'train.txt')
    val_txt= os.path.join(dataset_base_path, 'val.txt')
    test_txt = os.path.join(dataset_base_path, 'ImageSets/Main')
    # 数据增强
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
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
    learning_rate = 1e-3
    pretrained = False
    pretrain_weight = './yolov4/checkpoints/yolo4tf2_weight.pth'
    save_weight = 'voc.pth'
    anchors_path = './yolov4/data/yolo_anchors.txt'
    # Inference
    predict_weight = './model/yolo4_voc_weights.pth'
    ISTINY=False
    ATTENTION=0
    score=0.3
    iou=0.5
    imagesize=512
    # 标签平滑。一般0.01以下。如0.01、0.005。
    label_smoothing = 0
    # ciou or siou
    iou_type = 'ciou'
    save_period = 10
    # Env
    cuda =False
    '''
      distributed     用于指定是否使用单机多卡分布式运行
                      终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
                      Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
      DP模式：
          设置            distributed = False
          在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
      DDP模式：
          设置            distributed = True
          在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    '''
    distributed = False

