# train
logdir = './logs/'
annotation_path = './villages/train.txt'
classes_path='./village/village.names'
nchors_path = './model_data/yolo_anchors.txt'
pretrain_weight = './model_data/yolo4tf2_weight.h5'
save_model_name = 'village_tf2.h5'
imagesize=416
eager=False
'''
是否对损失进行归一化，用于改变loss的大小
用于决定计算最终loss是除上batch_size还是除上正样本数量
'''
normalize = False
'''
Yolov4的tricks应用
实际测试时mosaic数据增强并不稳定，所以默认为False
Cosine_scheduler 余弦退火学习率 True or False
label_smoothing 标签平滑 0.01以下一般 如0.01、0.005
'''
mosaic = False
Cosine_scheduler = False
label_smoothing = 0

regularization = True
'''
Init_epoch：初始训练世代
Freeze_epoch：冻结训练世代
epoch：最终训练世代
冻结训练：Init_epoch~Freeze_epoch
解冻训练：Freeze_epoch~epoch
'''
Init_epoch          = 0
Freeze_epoch        = 50
epoch = 100
freeze_layers = 249
batch_size          = 2
learning_rate_freeze  = 1e-3
learning_rate_unfreeze  = 1e-4

# predict
model_path='./model_data/village_tf2.h5'
anchors_path='./model_data/yolo_anchors.txt'
score=0.02
iou=0.3
max_boxes=100
letterbox_image=False
onnx=False
mode = "predict"
image = './test.png'
test_txt_file = './villages/test.txt'

video_path      = 0
video_save_path = ""
video_fps       = 25.0