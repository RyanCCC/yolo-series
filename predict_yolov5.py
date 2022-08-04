import time
import cv2
import tensorflow as tf
import colorsys
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from PIL import Image, ImageFont, ImageDraw
from yolov5.net import yolo_body
from utils.dataloader_yolov5 import get_anchors, get_classes, cvtColor
from yolov5.tools import DecodeBox
import Customerconfig
import os
import numpy as np



class YOLO(object):
    _defaults = {
        "model_path" : Customerconfig.model_path,
        "classes_path" : Customerconfig.classes_path,
        "anchors_path" : Customerconfig.anchors_path,
        "anchors_mask" : Customerconfig.ANCHOR_MASK,
        "input_shape" : [640, 640],
        "phi" : 's',
        "confidence" : 0.7,
        "nms_iou" : 0.5,
        "max_boxes": 100,
        "letterbox_image":True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 

        self.class_names = get_classes(self.classes_path)
        self.num_classes = len(self.class_names)
        self.anchors = get_anchors(self.anchors_path)
        self.num_anchors = len(self.anchors)

        # 设置颜色
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.get_predict_model()

    # 加载模型
    def get_predict_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi)
        self.model.load_weights(self.model_path, skip_mismatch=True, by_name=True)
        print('{} model, anchors, and classes loaded.'.format(model_path))
        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'anchors'           : self.anchors, 
                'num_classes'       : self.num_classes, 
                'input_shape'       : self.input_shape, 
                'anchor_mask'       : self.anchors_mask,
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes
    
    def preprocess_input(self, image):
        image /= 255.
        return image
    
    def resize_image(self, image, size, letterbox_image):
        iw, ih = image.size
        w, h = size
        if letterbox_image:
            scale = min(w/iw, h/ih)
            nw  = int(iw*scale)
            nh = int(ih*scale)

            image   = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    def detect_image(self, image, crop = False, count = False):
        image = cvtColor(image)

        image_data  = self.resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(self.preprocess_input(np.array(image_data, dtype='float32')), 0)

        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font        = ImageFont.truetype(font='./data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        if count:
            print("top_label:", out_classes)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(out_classes == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        if crop:
            for i, c in list(enumerate(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
    
if __name__ == '__main__':
    yolo = YOLO()
    crop = False
    count = False
    img = './samples/20210803173302.jpg'
    image = Image.open(img)
    r_image = yolo.detect_image(image, crop = crop, count=count)
    r_image.show()
