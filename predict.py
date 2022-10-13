from customerConf import *
from PIL import Image
from glob import glob
from yolox import Inference_YOLOXModel


if __name__=='__main__':
    path_pattern = './samples/*'
    yolox = Inference_YOLOXModel(YOLOXConfig)
    letterbox_image = True
    for path in glob(path_pattern):
        image = Image.open(path)
        img = yolox.detect(image)
        img.show()
    print('finish')