  
import colorsys
import os
import time
import config as sys_config
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from utils.utils import letterbox_image
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

if not sys_config.ISTINY:
    from nets.yolo4 import yolo_body, yolo_eval, yolo_head,yolo_correct_boxes, yolo_boxes_and_scores
else:
    from nets.yolo4_tiny import yolo_body, yolo_eval, yolo_head, yolo_correct_boxes, yolo_boxes_and_scores


class YOLO(object):

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"
    def __init__(self, **kwargs):
        self._defaults={
            "model_path"       : kwargs['model_path'],
            "anchors_path"      : kwargs['anchors_path'],
            "classes_path"      : kwargs['classes_path'],
            "score"             : kwargs['score'],
            "iou"               : kwargs['iou'],
            "max_boxes"         : kwargs['max_boxes'],
            "model_image_size"  : kwargs['model_image_size'],
            "letterbox_image"   : kwargs['letterbox_image'],
        }
        self.__dict__.update(self._defaults)
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.generate()

    #---------------------------------------------------#
    #   获得所有的分类
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names


    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        if not sys_config.ISTINY:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes, phi=sys_config.ATTENTION)
        else:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes, phi=sys_config.ATTENTION)
        self.yolo_model.load_weights(self.model_path)
        # self.yolo_model.save('./village_model', save_format='tf')
        # self.yolo_model1 = tf.keras.models.load_model('./village_model')
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        # 打乱颜色
        np.random.seed(10101)
        np.random.shuffle(self.colors)
        np.random.seed(None)


        self.input_image_shape = Input([2,],batch_size=1)
        inputs = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(yolo_eval, output_shape=(1,), name='yolo_eval',
            arguments={'anchors': self.anchors, 'num_classes': len(self.class_names), 'image_shape': self.model_image_size, 
            'score_threshold': self.score, 'eager': True, 'max_boxes': self.max_boxes, 'letterbox_image': self.letterbox_image})(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)
        self.yolo_model.summary()
        print('debug')
 
    @tf.function
    def get_pred_1(self, image_data):
        yolo_outputs = self.yolo_model1([image_data], training=False)
        return yolo_outputs
    
    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes


    def detect_image(self, image):

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
        # yolo_outputs = self.get_pred_1(image_data)
        # # 推理后处理
        # image_shape1 = K.reshape(yolo_outputs[-1], [-1])
        # num_layers = len(yolo_outputs)
        # anchor_mask = config.ANCHOR_MASK
        # input_shape = K.shape(yolo_outputs[0])[1:3] * 32
        # boxes = []
        # box_scores = []

        # for l in range(num_layers):
        #     _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], self.anchors[anchor_mask[l]], len(self.class_names), input_shape,
        #                                             image_shape, letterbox_image)
        #     boxes.append(_boxes)
        #     box_scores.append(_box_scores)
        # boxes = K.concatenate(boxes, axis=0)
        # box_scores = K.concatenate(box_scores, axis=0)
        # mask = box_scores >= self.score
        # max_boxes_tensor = K.constant(self.max_boxes, dtype='int32')
        # boxes_ = []
        # scores_ = []
        # classes_ = []
        # for c in range(len(self.class_names)):
        #     # 对小目标类别的阈值调低一些
        #     class_boxes = tf.boolean_mask(boxes, mask[:, c])
        #     class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #     # tensorflow非极大值抑制：https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
        #     nms_index = tf.image.non_max_suppression(
        #         class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=0.5)
        # # K.gather 检索张量reference中索引indices的元素
        #     class_boxes = K.gather(class_boxes, nms_index)
        #     class_box_scores = K.gather(class_box_scores, nms_index)
        #     classes = K.ones_like(class_box_scores, 'int32') * c
        #     boxes_.append(class_boxes)
        #     scores_.append(class_box_scores)
        #     classes_.append(classes)
        # out_boxes = K.concatenate(boxes_, axis=0)
        # out_scores = K.concatenate(scores_, axis=0)
        # out_classes = K.concatenate(classes_, axis=0)
        
        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = max((image.size[0] + image.size[1]) // 300, 1)
        
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
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
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image
    
    def deep_sort_track(self, image):
        
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
        boxes = []
        for i, c in list(enumerate(out_classes)):
            box_tmp = []
            predicted_class = self.class_names[c]
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

            # change left top right bottom to center_x, center_y, width, height
            width = right-left
            height = bottom-top
            box_tmp.append(left)
            box_tmp.append(top)
            box_tmp.append(width)
            box_tmp.append(height)
            
            boxes.append(box_tmp)
        return boxes, out_scores, out_classes
    
    def get_dr_txt(self, image_id,  image):
        
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
        
        # print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        
        if len(out_boxes) == 0:
            print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
            dr_txt_path = os.path.join(sys_config.result, sys_config.pr_folder_name, image_id+'.txt')
            with open(dr_txt_path, 'w') as f:
                f.write(" ")

        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[c]
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

            dr_txt_path = os.path.join(sys_config.result, sys_config.pr_folder_name, image_id+'.txt')
            with open(dr_txt_path, 'w') as f:
                f.write("%s %s %s %s %s %s\n" % (predicted_class, str(score.numpy()), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))
            

        return image

    def get_FPS(self, image, test_interval):

        if self.letterbox_image:
            boxed_image = letterbox_image(image, (self.model_image_size[1],self.model_image_size[0]))
        else:
            boxed_image = image.convert('RGB')
            boxed_image = boxed_image.resize((self.model_image_size[1],self.model_image_size[0]), Image.BICUBIC)
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        #---------------------------------------------------------#
        #   将图像输入网络当中进行预测！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time