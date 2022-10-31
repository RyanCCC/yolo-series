from ast import arg
import os
import sys
from tkinter.tix import Tree
sys.path.append(os.getcwd())
from cfg import *
from PIL import Image
from tqdm import tqdm
import argparse


'''
如果想要设定mAP0.x，比如设定mAP0.75，可以去config.py设定MINOVERLAP。
'''

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', 
        help='YOLOV4, YOLOV4-TINY, YOLOV5 or YOLOX', 
        choices=['YOLOV4', 'YOLOV4-TINY', 'YOLOV5', 'YOLOX', 'YOLOV7', 'YOLOV7-TINY'],
        default='YOLOX', 
        type=str)
    parser.add_argument(
        '--testset',
        help = 'testset file',
        type = str
    )

    parser.add_argument(
        '--pr_folder',
        help='prediction save path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--model_path',
        help = 'model weight path',
        required=True
    )

    parser.add_argument(
        '--image_path',
        help='image path',
        required=True
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    if args.model.upper() == 'YOLOX':
        from yolox import Inference_YOLOXModel
        yolo = Inference_YOLOXModel(YOLOXConfig, args.model_path)
    elif  args.model.upper() == 'YOLOV4':
        from yolov4 import Inference_YOLOV4Model
        yolo = Inference_YOLOV4Model(YOLOV4Config, args.model_path)
    elif args.model.upper() == 'YOLOV4-TINY':
        from yolov4 import Inference_YOLOV4Model
        yolo = Inference_YOLOV4Model(YOLOV4Config, args.model_path)
    elif args.model.upper() == 'YOLOV5':
        from yolov5 import Inference_YOLOV5Model
        yolo = Inference_YOLOV5Model(YOLOV5Config, args.model_path)
    elif args.model.upper() == 'YOLOV7':
        from yolov7 import Inference_YOLOV7Model
        yolo = Inference_YOLOV7Model(YOLOV7Config, args.model_path)
    elif args.model.upper() == 'YOLOV7-TINY':
        from yolov7 import Inference_YOLOV7Model
        yolo = Inference_YOLOV7Model(YOLOV7Config, args.model_path)
    else:
        pass

    image_ids = open(args.testset).read().strip().split()
    pr_folder_name = args.pr_folder
    image_path = args.image_path
    if not os.path.exists(pr_folder_name):
        os.makedirs(pr_folder_name)

    for image_id in tqdm(image_ids):
        image_name = os.path.join(image_path, image_id+".jpg")
        image = Image.open(image_name)
        yolo.getdrtxt(image, pr_folder_name, image_id)
    
    print("Conversion completed!")
