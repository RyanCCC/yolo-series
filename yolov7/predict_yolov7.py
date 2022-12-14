import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import colorsys
import os
import time
import gc

import numpy as np
import tensorflow as tf
from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from .nets import yolo_body, fusion_rep_vgg
from .lib.tools import cvtColor, get_anchors, get_classes, preprocess_input,resize_image, show_config
from .lib.decodebox import DecodeBox


class YOLO(object):
    def __init__(self, **kwargs):
        self._params = {
            "model_path" : kwargs['model_path'], 
            "classes_path" : kwargs['class_path'], 
            "anchors_path" : kwargs['anchors_path'], 
            "anchors_mask" : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            "input_shape" : kwargs['input_shape'],
            "phi" : kwargs['phi'],
            "confidence" : kwargs['confidence'], 
            "nms_iou" : kwargs['nms_iou'],
            "max_boxes" : kwargs['max_boxes'], 
            "letterbox_image" : kwargs['letterbox_image'],
            "tiny":kwargs['tiny'],
            "onnx":kwargs['onnx']
        }
        self.__dict__.update(self._params)
            
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.init_model()

    def init_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi)
        self.model.load_weights(self.model_path, by_name=True)
        
        if self.phi == "l":
            fuse_layers = [
                ["rep_conv_1", False, True],
                ["rep_conv_2", False, True],
                ["rep_conv_3", False, True],
            ]
            self.model_fuse = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi, mode="predict")
            self.model_fuse.load_weights(self.model_path, by_name=True)

            fusion_rep_vgg(fuse_layers, self.model, self.model_fuse)
            del self.model
            gc.collect()
            self.model = self.model_fuse
        print('{} model, anchors, and classes loaded.'.format(model_path))
        if self.onnx:
            return
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

    def detect(self, image, crop = False, istrack=False):
        image = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        
        if crop:
            for i, c in list(enumerate(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
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
            font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
            thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
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
                    draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
                draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
                draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
                del draw

            return image

    def getdrtxt(self,image, pr_folder_name, image_id):
        image = cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
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

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        image = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        output  = self.model.predict(image_data)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in output:
            b, h, w, c = np.shape(sub_output)
            sub_output = np.reshape(sub_output, [b, h, w, 3, -1])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()


def Inference_YOLOV7Model(config, model_path, onnx = False):
    yolo = YOLO(
        model_path = model_path,
        class_path = config.classes_path,
        anchors_path = config.anchor_path,
        anchors_mask = config.ANCHOR_MASK,
        input_shape = config.input_shape,
        confidence = config.score,
        nms_iou = config.iou,
        max_boxes=config.max_boxes,
        letterbox_image = True,
        phi=config.phi,
        tiny = config.tiny,
        onnx = False
    )
    return yolo
