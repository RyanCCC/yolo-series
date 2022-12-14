import os
from cfg import *
from PIL import Image
from glob import glob
import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument(
        '--model', 
        help='YOLOV4, YOLOV4-TINY, YOLOV5 or YOLOX', 
        choices=['YOLOV4', 'YOLOV4-TINY', 'YOLOV5','YOLOV5-V61', 'YOLOX', 'YOLOV7', 'YOLOV7-TINY'],
        default='YOLOV5', 
        type=str)
    parser.add_argument('--show', action='store_true', help='show preidict result. Not recommended')
    parser.add_argument('--save', action='store_true', help='save result image.')
    parser.add_argument('--img_dir', default='./samples', help='predict image dir')
    parser.add_argument('--weights', help='model weights', required=True)
    parser.add_argument('--save_dir', default='./result', help='save_dir')
    parser.add_argument('--source', help='source,image file/dir')
    args = parser.parse_args()
    return args

def dir_inference(imag_dir, model, args):
    path_pattern = f'{imag_dir}/*'
    img_number = len(path_pattern)
    result = 0
    for path in glob(path_pattern):
        # 判断是否为文件夹
        if os.path.isdir(path):
            continue
        image = Image.open(path)
        start_time = time.time()
        img = model.detect(image)
        end_time = time.time()
        result += (end_time-start_time)
        if args.show:
            img.show()
        if args.save:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, os.path.basename(path))
            img.save(save_path)
    fps = 1/(result/img_number)
    print(f'finish，fps is {fps}')

if __name__=='__main__':
    args = parse_args()
    source = args.source
    if args.model.upper() == 'YOLOX':
        from yolox import Inference_YOLOXModel
        yolo = Inference_YOLOXModel(YOLOXConfig, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)
    elif  args.model.upper() == 'YOLOV4':
        from yolov4 import Inference_YOLOV4Model
        yolo = Inference_YOLOV4Model(YOLOV4Config, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)
    elif args.model.upper() == 'YOLOV4-TINY':
        from yolov4 import Inference_YOLOV4Model
        yolo = Inference_YOLOV4Model(YOLOV4Config, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)
    elif args.model.upper() == 'YOLOV5':
        from yolov5 import Inference_YOLOV5Model
        yolo = Inference_YOLOV5Model(YOLOV5Config, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)
    
    elif args.model.upper() == 'YOLOV5-V61':
        from yolov5v61 import Inference_YOLOV5Model
        yolo = Inference_YOLOV5Model(YOLOV5Config, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)

    elif args.model.upper() == 'YOLOV7':
        from yolov7 import Inference_YOLOV7Model
        yolo = Inference_YOLOV7Model(YOLOV7Config, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)
    elif args.model.upper() == 'YOLOV7-TINY':
        from yolov7 import Inference_YOLOV7Model
        yolo = Inference_YOLOV7Model(YOLOV7Config, args.weights)
        if not os.path.isfile(source):
            dir_inference(args.img_dir, yolo, args)
        else:
            image = Image.open(source)
            img = yolo.detect(image)
            if args.show:
                img.show()
            if args.save:
                save_path = os.path.join(args.save_dir, 'tmp.jpg')
                img.save(save_path)
    else:
        pass
    # path_pattern = './samples/*'
    # from yolov5 import Inference_YOLOV5Model
    # # yolox = Inference_YOLOXModel(YOLOXConfig)
    # # letterbox_image = True
    # # for path in glob(path_pattern):
    # #     image = Image.open(path)
    # #     img = yolox.detect(image)
    # #     img.show()
    # # print('finish yolox')
    # # yolov4 = Inference_YOLOV4Model(YOLOV4Config, './model/village2022_yolov4_20221013.h5')
    # # letterbox_image = True
    # # for path in glob(path_pattern):
    # #     image = Image.open(path)
    # #     img = yolov4.inference(image)
    # #     img.show()
    # # print('finish')
    # yolov5 = Inference_YOLOV5Model(YOLOV5Config, './model/village2022_yolov5_l_20221013.h5')
    # for path in glob(path_pattern):
    #     image = Image.open(path)
    #     img = yolov5.detect_image(image)
    #     img.show()
    # print('finish')