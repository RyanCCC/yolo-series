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
    cuda = False
    gpus = '1'
    '''
    单机多卡分布式训练
    DP模式：
    设置 distributed = False
    在终端中输入 CUDA_VISIBLE_DEVICES=0,1 python train.py
    DDP模式：
    设置 distributed = True
    在终端中输入 CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    '''
    distributed = False
    # DDP模式多卡可用
    sync_bn = False
    fp16 = False
    # 标签平滑。一般0.01以下。如0.01、0.005
    label_smoothing = 0
    pretrain_weight = './yolov7/checkpoints/yolov7_x_weights.pth'
    save_weight = 'village_yolov7_x_20221019.pth'
    # 0:DEBUG；1：INFO；2：warning；3：error
    log_level = '1'
    Freeze_Train = True
    learning_rate = 1e-2


