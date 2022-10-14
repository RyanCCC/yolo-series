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
    score=0.3
    iou=0.5
    max_boxes=100
    letterbox_image=False
    ANCHOR_MASK = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]