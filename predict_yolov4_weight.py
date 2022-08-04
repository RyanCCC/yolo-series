import colorsys
import os
import time
from Customerconfig import YOLOV4Config
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

if not YOLOV4Config.ISTINY:
    from nets.yolo4 import yolo_body, yolo_eval
else:
    from nets.yolo4_tiny import yolo_body, yolo_eval
from utils.utils import letterbox_image

class YOLOV4(object):
    def __init__(self, **kwargs) -> None:
        self._params={
            "model_path"       : kwargs['model_path'],
            "anchors_path"      : kwargs['anchors_path'],
            "classes_path"      : kwargs['classes_path'],
            "score"             : kwargs['score'],
            "iou"               : kwargs['iou'],
            "max_boxes"         : kwargs['max_boxes'],
            "model_image_size"  : kwargs['model_image_size'],
            "letterbox_image"   : kwargs['letterbox_image'],
        }
        self.__dict__.update(self._params)
        self._class_names = self.get_classes()
        self._anchors = self.get_anchors()
        # 初始化颜色
        hsv_tuples = [(x / len(self._class_names), 1., 1.)
                      for x in range(len(self._class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)
        # 初始化模型
        self.yolo_model = self.get_model()
    
    # 获得所有分类
    def get_classes(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path,encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names
    
    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)
    
    def get_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        num_anchors = len(self._anchors)
        num_classes = len(self._class_names)
        if not YOLOV4Config.ISTINY:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes, phi=YOLOV4Config.ATTENTION)
        else:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, phi=YOLOV4Config.ATTENTION)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        
        self.input_image_shape = Input([2,],batch_size=1)
        inputs = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
            arguments={'anchors': self._anchors, 'num_classes': len(self._class_names), 'image_shape': self.model_image_size, 
            'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes, 'letterbox_image': self.letterbox_image})(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        return self.yolo_model
    
    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes
    
    def inference(self, image,istrack=False, isdrtxt=False, image_id=None):
        '''
        参数说明：
        image：待检测的图像
        imageid：在计算map的时候需要用到
        istrack：是否目标跟踪返回数据的标志
        isdrtxt：是否计算map的txt标志
        '''
        image = image.convert('RGB')
        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        if istrack:
            boxes = []
            for i, c in list(enumerate(out_classes)):
                box_tmp = []
                box = out_boxes[i]
                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
                # change left top right bottom to center_x, center_y, width, height
                width = right-left
                height = bottom-top
                box_tmp.append(left)
                box_tmp.append(top)
                box_tmp.append(width)
                box_tmp.append(height)
                boxes.append(box_tmp)
            return boxes, out_scores, out_classes
        elif isdrtxt:
            if len(out_boxes) == 0:
                print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
                dr_txt_path = os.path.join(YOLOV4Config.result, YOLOV4Config.pr_folder_name, image_id+'.txt')
                with open(dr_txt_path, 'w') as f:
                    f.write(" ")

            for i, c in list(enumerate(out_classes)):
                predicted_class = self._class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                dr_txt_path = os.path.join(YOLOV4Config.result, YOLOV4Config.pr_folder_name, image_id+'.txt')
                with open(dr_txt_path, 'w') as f:
                    f.write("%s %s %s %s %s %s\n" % (predicted_class, str(score.numpy()), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
        else:
            font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1]*2 + 0.5).astype('int32'))
            thickness = max((image.size[0] + image.size[1]) // 300, 1)
        
            for i, c in list(enumerate(out_classes)):
                predicted_class = self._class_names[c]
                box = out_boxes[i]
                score = out_scores[i]

                top, left, bottom, right = box
                top = top - 5
                left = left - 5
                bottom = bottom + 5
                right = right + 5
                top = max(0, np.floor(top + 0.5).astype('int32'))
                left = max(0, np.floor(left + 0.5).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
                right = min(image.size[0], np.floor(right + 0.5).astype('int32'))

                # 画框框
                label = '{} {:.2f}'.format(predicted_class, score)
                # label = '{}'.format(predicted_class)
                draw = ImageDraw.Draw(image)
                label_size = draw.textsize(label, font)
                label = label.encode('utf-8')
                print(label, top, left, bottom, right)
            
                if top - label_size[1] >= 0:
                    text_origin = np.array([left, top - label_size[1]])
                else:
                    text_origin = np.array([left, top + 1])

                for i in range(thickness):
                    draw.rectangle([left + i, top + i, right - i, bottom - i],outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

            return image



        

    

