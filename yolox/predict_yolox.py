import colorsys
import os
import sys 
from cfg import YOLOXConfig
import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont,Image
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from .nets.yolox import yolo_body
from .lib.dataloader import cvtColor, get_classes, preprocess_input
import tensorflow.keras.backend as K
import gc
from glob import glob

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset = (input_shape - new_shape)/2./input_shape
        scale = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def DecodeBox(outputs, num_classes, input_shape, max_boxes = 100, confidence=0.5, nms_iou=0.3, letterbox_image=True):
    image_shape = K.reshape(outputs[-1], [-1])
    outputs = outputs[:-1]
    batch_size = K.shape(outputs[0])[0]
    grids = []
    strides = []
    hw = [K.shape(x)[1:3] for x in outputs]
    outputs = tf.concat([tf.reshape(x, [batch_size, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        grid_x, grid_y  = tf.meshgrid(tf.range(hw[i][1]), tf.range(hw[i][0]))
        grid = tf.reshape(tf.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape = tf.shape(grid)[:2]
        grids.append(tf.cast(grid, K.dtype(outputs)))
        strides.append(tf.ones((shape[0], shape[1], 1)) * input_shape[0] / tf.cast(hw[i][0], K.dtype(outputs)))
    grids = tf.concat(grids, axis=1)
    strides = tf.concat(strides, axis=1)
    box_xy = (outputs[..., :2] + grids) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_wh = tf.exp(outputs[..., 2:4]) * strides / K.cast(input_shape[::-1], K.dtype(outputs))
    box_confidence  = K.sigmoid(outputs[..., 4:5])
    box_class_probs = K.sigmoid(outputs[..., 5: ])
    boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)
    box_scores  = box_confidence * box_class_probs

    mask = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[..., c])
        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out = K.concatenate(boxes_out, axis=0)
    scores_out = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


class YOLOX(object):
    def __init__(self, **kwargs) -> None:
        self._arguments = {
            'class_path':kwargs['class_path'], 
            'input_shape':kwargs['input_shape'], 
            'confidence':kwargs['confidence'], 
            'nms_iou':kwargs['nms_iou'], 
            'max_boxes':kwargs['max_boxes'], 
            'letterbox_image':kwargs['letterbox_image'],
            'model_path':kwargs['model_path'],
            'phi':kwargs['phi']
        }
        self.__dict__.update(self._arguments)
        self.class_names = get_classes(self.class_path)
        self.model = self.build_model()

    def resize_image(self, image):
        size = self.input_shape
        iw, ih  = image.size
        w, h    = size
        if self.letterbox_image:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)

            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', size, (128,128,128))
            new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        else:
            new_image = image.resize((w, h), Image.BICUBIC)
        return new_image

    def build_model(self, export_model = False):
        num_classes = len(self.class_names)
        yolo_model = yolo_body([None, None, 3], num_classes=num_classes, phi=self.phi)
        yolo_model.load_weights(self.model_path)
        print('model weight success load.')
        if export_model:
            yolo_model.save('./model/yolox_model', save_format='tf2')
            yolo_model = tf.keras.models.load_model('./model/yolox_model')
            print('model export success.')
        input_image_shape = Input([2, ], batch_size=1)
        inputs = [*yolo_model.output, input_image_shape]
        outputs = Lambda(
            DecodeBox,
            output_shape = (1, ), 
            name = 'yolo_eval',
            arguments = {
                'num_classes'       : num_classes, 
                'input_shape'       : self.input_shape, 
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
            }
        )(inputs)
        # if export_model:
        #     yolo_model.save('./model/yolox_model', save_format='tf2')
        #     yolo_model = tf.keras.models.load_model('./model/yolox_model', custom_objects={'yolo_eval':self.DecodeBox})
        model = Model([yolo_model.input, input_image_shape], outputs)
        gc.collect()
        return model

    @tf.function
    def prediction(self, model, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes

    def detect(self, image, crop=False, istrack=False):
        num_classes = len(self.class_names)
        # 设置颜色
        hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
        colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
        # 构建模型
        image = cvtColor(image)
        image_data = self.resize_image(image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.prediction(self.model, image_data, input_image_shape) 
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

        else:
            font = ImageFont.truetype(font='data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
            if crop:
                for i, c in list(enumerate(out_boxes)):
                    top, left, bottom, right = out_boxes[i]
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                    right = min(image.size[0], np.floor(right).astype('int32'))
                    dir_save_path = "img_crop"
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    crop_image = image.crop([left, top, right, bottom])
                    crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                    print("save crop_" + str(i) + ".png to " + dir_save_path)
            for i, c in list(enumerate(out_classes)):
                predicted_class = self.class_names[int(c)]
                box = out_boxes[i]
                score = out_scores[i]
                top, left, bottom, right = box

                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
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
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw
            return image
    def getdrtxt(self,image, pr_folder_name, image_id):
        image = cvtColor(image)
        image_data = self.resize_image(image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.prediction(self.model, image_data, input_image_shape)
        if len(out_boxes) == 0:
            dr_txt_path = os.path.join(pr_folder_name, image_id+'.txt')
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

            dr_txt_path = os.path.join(pr_folder_name, image_id+'.txt')
            with open(dr_txt_path, 'w') as f:
                f.write("%s %s %s %s %s %s\n" % (predicted_class, str(score.numpy()), str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

def Inference_YOLOXModel(YOLOXConfig, model_path = './model/village2022_yolox_s_20221017.h5'):
    yolox = YOLOX(
        class_path = YOLOXConfig.classes_path,
        input_shape = YOLOXConfig.input_shape,
        confidence = YOLOXConfig.score,
        nms_iou = YOLOXConfig.iou,
        max_boxes=YOLOXConfig.max_boxes,
        letterbox_image = True,
        model_path = model_path,
        phi=YOLOXConfig.phi
    )
    return yolox

# 创建yolox
# model_path = './model/village_yolox.h5'
# yolox = YOLOX(
#     class_path = YOLOXConfig.classes_path,
#     input_shape = YOLOXConfig.input_shape,
#     confidence = YOLOXConfig.score,
#     nms_iou = YOLOXConfig.iou,
#     max_boxes=YOLOXConfig.max_boxes,
#     letterbox_image = True,
#     model_path = model_path,
#     phi=YOLOXConfig.phi
# )
