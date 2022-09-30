import colorsys
import os
import time
import gc

import numpy as np
from keras import backend as K
from PIL import ImageDraw, ImageFont
from customerConf import YOLOXV7Config
from yolov7 import yolo_body, fusion_rep_vgg, cvtColor, get_anchors, get_classes, preprocess_input, resize_image, show_config, DecodeBox
from PIL import Image 
from glob import glob

class YOLO(object):
    def __init__(self, **kwargs):
        self._arguments = {
            "model_path"        : kwargs['model_path'], 
            "classes_path"      : kwargs['class_path'], 
            "anchors_path"      : kwargs['anchors_path'], 
            "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            "input_shape"       : kwargs['input_shape'],
            "phi"               : kwargs['phi'],
            "confidence"        : kwargs['confidence'], 
            "nms_iou"           : kwargs['nms_iou'],
            "max_boxes"         : kwargs['max_boxes'], 
            "letterbox_image"   : kwargs['letterbox_image']
        }

        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.input_image_shape = K.placeholder(shape=(2, ))

        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.init_model()

    def init_model(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.yolo_model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi)
        self.yolo_model.load_weights(self.model_path, by_name=True)
        if self.phi == "l":
            fuse_layers = [
                ["rep_conv_1", False, True],
                ["rep_conv_2", False, True],
                ["rep_conv_3", False, True],
            ]
            self.yolo_model_fuse = yolo_body([None, None, 3], self.anchors_mask, self.num_classes, self.phi, mode="predict")
            self.yolo_model_fuse.load_weights(self.model_path, by_name=True)

            fusion_rep_vgg(fuse_layers, self.yolo_model, self.yolo_model_fuse)
            del self.yolo_model
            gc.collect()
            self.yolo_model = self.yolo_model_fuse
        print('{} model, anchors, and classes loaded.'.format(model_path))
        boxes, scores, classes = DecodeBox(
            self.yolo_model.output, 
            self.anchors,
            self.num_classes, 
            self.input_image_shape, 
            self.input_shape, 
            anchor_mask     = self.anchors_mask,
            max_boxes       = self.max_boxes,
            confidence      = self.confidence, 
            nms_iou         = self.nms_iou, 
            letterbox_image = self.letterbox_image
        )
        return boxes, scores, classes


    def detect(self, image, crop = False, count = False):

        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0})

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        if count:
            print("top_label:", out_classes)
            classes_nums    = np.zeros([self.num_classes])
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


    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        image       = cvtColor(image)
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        output  = self.yolo_model.predict(image_data)
        
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
        
    def close_session(self):
        self.sess.close()

# 初始化模型
yolo = YOLO(
    class_path = YOLOXV7Config.classes_path,
    input_shape = YOLOXV7Config.input_shape,
    confidence = YOLOXV7Config.score,
    nms_iou = YOLOXV7Config.iou,
    max_boxes=YOLOXV7Config.max_boxes,
    letterbox_image = True,
    model_path = '',
    phi=YOLOXV7Config.phi
)

if __name__ == '__main__':
    path_pattern = './samples/*'
    
    letterbox_image = True
    for path in glob(path_pattern):
        image = Image.open(path)
        img = yolo.detect(image)
        img.show()
    print('finish')