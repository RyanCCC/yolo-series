LOG_DIR = 'runs/'
GPUS = (0,1)     
WORKERS = 8
PIN_MEMORY = False
PRINT_FREQ = 20
AUTO_RESUME =False       # Resume from the last training interrupt
NEED_AUTOANCHOR = False      # Re-select the prior anchor(k-means)    When training from scratch (epoch=0), set it to be ture!
DEBUG = False
num_seg_class = 2

# Cudnn related params
CUDNN_BENCHMARK = True
CUDNN_DETERMINISTIC = False
CUDNN_ENABLED = True


# common params for NETWORK
MODEL_NAME = ''
MODEL_STRU_WITHSHARE = False     #add share_block to segbranch
MODEL_HEADS_NAME = ['']
MODEL_PRETRAINED = ""
MODEL_PRETRAINED_DET = ""
MODEL_IMAGE_SIZE = [640, 640]  # width * height, ex: 192 * 256


# loss params
LOSS_LOSS_NAME = ''
LOSS_MULTI_HEAD_LAMBDA = None
LOSS_FL_GAMMA = 0.0  # focal loss gamma
LOSS_CLS_POS_WEIGHT = 1.0  # classification loss positive weights
LOSS_OBJ_POS_WEIGHT = 1.0  # object loss positive weights
LOSS_SEG_POS_WEIGHT = 1.0  # segmentation loss positive weights
LOSS_BOX_GAIN = 0.05  # box loss gain
LOSS_CLS_GAIN = 0.5  # classification loss gain
LOSS_OBJ_GAIN = 1.0  # object loss gain
LOSS_DA_SEG_GAIN = 0.2  # driving area segmentation loss gain
LOSS_LL_SEG_GAIN = 0.2  # lane line segmentation loss gain
LOSS_LL_IOU_GAIN = 0.2 # lane line iou loss gain


# DATASET related params
DATASET_DATAROOT = '/home/bdd/bdd100k/images/100k'       # the path of images folder
DATASET_LABELROOT = '/home/bdd/bdd100k/labels/100k'      # the path of det_annotations folder
DATASET_MASKROOT = '/home/bdd/bdd_seg_gt'                # the path of da_seg_annotations folder
DATASET_LANEROOT = '/home/bdd/bdd_lane_gt'               # the path of ll_seg_annotations folder
DATASET_DATASET = 'BddDataset'
DATASET_TRAIN_SET = 'train'
DATASET_TEST_SET = 'val'
DATASET_DATA_FORMAT = 'jpg'
DATASET_SELECT_DATA = False
DATASET_ORG_IMG_SIZE = [720, 1280]

# training data augmentation
DATASET_FLIP = True
DATASET_SCALE_FACTOR = 0.25
DATASET_ROT_FACTOR = 10
DATASET_TRANSLATE = 0.1
DATASET_SHEAR = 0.0
DATASET_COLOR_RGB = False
DATASET_HSV_H = 0.015  # image HSV-Hue augmentation (fraction)
DATASET_HSV_S = 0.7  # image HSV-Saturation augmentation (fraction)
DATASET_HSV_V = 0.4  # image HSV-Value augmentation (fraction)
# TODO: more augmet params to add


# train
TRAIN_LR0 = 0.001  # initial learning rate (SGD=1E-2, Adam=1E-3)
TRAIN_LRF = 0.2  # final OneCycleLR learning rate (lr0 * lrf)
TRAIN_WARMUP_EPOCHS = 3.0
TRAIN_WARMUP_BIASE_LR = 0.1
TRAIN_WARMUP_MOMENTUM = 0.8

TRAIN_OPTIMIZER = 'adam'
TRAIN_MOMENTUM = 0.937
TRAIN_WD = 0.0005
TRAIN_NESTEROV = True
TRAIN_GAMMA1 = 0.99
TRAIN_GAMMA2 = 0.0

TRAIN_BEGIN_EPOCH = 0
TRAIN_END_EPOCH = 240

TRAIN_VAL_FREQ = 1
TRAIN_BATCH_SIZE_PER_GPU =24
TRAIN_SHUFFLE = True

TRAIN_IOU_THRESHOLD = 0.2
TRAIN_ANCHOR_THRESHOLD = 4.0

# if training 3 tasks end-to-end, set all parameters as True
# Alternating optimization
TRAIN_SEG_ONLY = False           # Only train two segmentation branchs
TRAIN_DET_ONLY = False           # Only train detection branch
TRAIN_ENC_SEG_ONLY = False       # Only train encoder and two segmentation branchs
TRAIN_ENC_DET_ONLY = False       # Only train encoder and detection branch

# Single task 
TRAIN_DRIVABLE_ONLY = False      # Only train da_segmentation task
TRAIN_LANE_ONLY = False          # Only train ll_segmentation task
TRAIN_DET_ONLY = False          # Only train detection task




TRAIN_PLOT = True                # 

# testing
TEST_BATCH_SIZE_PER_GPU = 24
TEST_MODEL_FILE = ''
TEST_SAVE_JSON = False
TEST_SAVE_TXT = False
TEST_PLOTS = True
TEST_NMS_CONF_THRESHOLD  = 0.001
TEST_NMS_IOU_THRESHOLD  = 0.6


def update_config(args):

    if args.modelDir:
        OUTPUT_DIR = args.modelDir

    if args.logDir:
        LOG_DIR = args.logDir
