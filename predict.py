'''
YOLOV5推理

Usage：
    python ./predict.py --model YOLOV7 --source ./samples/ --weights ./model/voc_2007.pt --save --save_dir ./result/ 
    
    python ./predict.py --model YOLOV5-V61 --source ./samples/images/1.jpg --weights ./model/voc_2007.onnx --save --save_dir ./result/

'''
import os
from cfg import *
from PIL import Image
from glob import glob
import argparse
import time
import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Custom Input')
    parser.add_argument(
        '--model', 
        help='YOLOV4, YOLOV4-TINY, YOLOV5 or YOLOX', 
        choices=['YOLOV4', 'YOLOV4-TINY', 'YOLOV5','YOLOV5-V61', 'YOLOX', 'YOLOV7', 'YOLOV7-TINY'],
        default='YOLOV5', 
        type=str)
    parser.add_argument('--save', action='store_true', help='save result image.')
    parser.add_argument('--weights', help='model weights', required=True)
    parser.add_argument('--save_dir', default='./result', help='save_dir')
    parser.add_argument('--source', help='source: image, dir, video or camera')
    return parser

def video_inference(src, model, save):
    capture = cv2.VideoCapture(src)
    fps = 0.0
    # 定义编解码器并创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out = cv2.VideoWriter('./output.mp4', fourcc, 20.0, size)
    while True:
        try:
            t1 = time.time()
            if capture.isOpened():
                ref, frame = capture.read()
                # 获取视频的时间戳
                millseconds = capture.get(cv2.CAP_PROP_POS_MSEC)
                if frame is not None and ref:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(np.uint8(frame))
                    frame = np.array(model.detect(frame))
                    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

                    fps  = ( fps + (1./(time.time()-t1)) ) / 2
                    print("fps= %.2f"%(fps))
                    frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if save:
                        out.write(frame)
                    cv2.imshow("video",frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
            else:
                break
        except Exception as e:
            print(e)
            break

    capture.release()
    out.release()
    cv2.destroyAllWindows()

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
        if args.save:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            save_path = os.path.join(args.save_dir, os.path.basename(path))
            img.save(save_path)
    fps = 1/(result/img_number)
    print(f'finish，fps is {fps}')

if __name__=='__main__':
    parser = parse_args()
    args = parser.parse_args()
    source = args.source
    webcam = source.isnumeric() or source.lower().endswith(('.mp4', '.mp3', '.avi')) or source.lower().startswith(('rtsp://', 'rtmp://'))
    if args.model.upper() == 'YOLOX':
        from yolox import Inference_YOLOXModel
        yolo = Inference_YOLOXModel(YOLOXConfig, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)
    elif  args.model.upper() == 'YOLOV4':
        from yolov4 import Inference_YOLOV4Model
        yolo = Inference_YOLOV4Model(YOLOV4Config, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)
    elif args.model.upper() == 'YOLOV4-TINY':
        from yolov4 import Inference_YOLOV4Model
        yolo = Inference_YOLOV4Model(YOLOV4Config, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)
    elif args.model.upper() == 'YOLOV5':
        from yolov5 import Inference_YOLOV5Model
        yolo = Inference_YOLOV5Model(YOLOV5Config, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)
    
    elif args.model.upper() == 'YOLOV5-V61':
        from yolov5v61 import Inference_YOLOV5Model
        yolo = Inference_YOLOV5Model(YOLOV5Config, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)

    elif args.model.upper() == 'YOLOV7':
        from yolov7 import Inference_YOLOV7Model
        yolo = Inference_YOLOV7Model(YOLOV7Config, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)
    elif args.model.upper() == 'YOLOV7-TINY':
        from yolov7 import Inference_YOLOV7Model
        yolo = Inference_YOLOV7Model(YOLOV7Config, args.weights)
        if webcam:
            video_inference(source, yolo, args.save)
        else:
            if os.path.isdir(source):
                dir_inference(args.source, yolo, args)
            else:
                image = Image.open(source)
                img = yolo.detect(image)
                img.show()
                if args.save:
                    save_path = os.path.join(args.save_dir, 'tmp.jpg')
                    img.save(save_path)
    else:
        pass