from cfg import *
from PIL import Image
from glob import glob
from yolox import Inference_YOLOXModel
from yolov4 import Inference_YOLOV4Model
from yolov5 import Inference_YOLOV5Model



if __name__=='__main__':
    path_pattern = './samples/*'
    # yolox = Inference_YOLOXModel(YOLOXConfig)
    # letterbox_image = True
    # for path in glob(path_pattern):
    #     image = Image.open(path)
    #     img = yolox.detect(image)
    #     img.show()
    # print('finish yolox')
    # yolov4 = Inference_YOLOV4Model(YOLOV4Config, './model/village2022_yolov4_20221013.h5')
    # letterbox_image = True
    # for path in glob(path_pattern):
    #     image = Image.open(path)
    #     img = yolov4.inference(image)
    #     img.show()
    # print('finish')
    yolov5 = Inference_YOLOV5Model(YOLOV5Config, './model/village2022_yolov5_l_20221013.h5')
    letterbox_image = True
    for path in glob(path_pattern):
        image = Image.open(path)
        img = yolov5.detect_image(image)
        img.show()
    print('finish')