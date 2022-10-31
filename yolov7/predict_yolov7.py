import colorsys
import os
from turtle import width
import torch
import torch.nn as nn
import numpy as np
from PIL import ImageDraw, ImageFont
from yolov7 import YoloBody, cvtColor, get_anchors, get_classes, preprocess_input, resize_image, DecodeBox
from yolov7 import yoloBodyTiny
from PIL import Image 
from glob import glob

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
            "tiny":kwargs['tiny']
        }
        self.__dict__.update(self._params)
        self.cuda = False
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors = get_anchors(self.anchors_path)
        self.bbox_util = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.init_model()


    def init_model(self, onnx=False):
        if self.tiny:
            self.net = yoloBodyTiny(self.anchors_mask, self.num_classes)
        else:
            self.net = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()


    def detect(self, image, crop = False, count = False, istrack = False, isdrtxt = False, image_id = None):
        image_shape = np.array(np.shape(image)[0:2])
        image= cvtColor(image)
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label = np.array(results[0][:, 6], dtype = 'int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        font = ImageFont.truetype(font='./font/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        if istrack:
            boxes = []
            if count:
                print("top_label:", top_label)
                classes_nums = np.zeros([self.num_classes])
                for i in range(self.num_classes):
                    num = np.sum(top_label == i)
                    if num > 0:
                        print(self.class_names[i], " : ", num)
                    classes_nums[i] = num
                print("classes_nums:", classes_nums)
            if crop:
                for i, c in list(enumerate(top_boxes)):
                    top, left, bottom, right = top_boxes[i]
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                    right = min(image.size[0], np.floor(right).astype('int32'))
                    dir_save_path = "img_crop"
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    crop_image = image.crop([left, top, right, bottom])
                    crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                    print("save crop_" + str(i) + ".png to " + dir_save_path)
            for i, c in list(enumerate(top_label)):
                box_tmp = []
                predicted_class = self.class_names[int(c)]
                box = top_boxes[i]
                score = top_conf[i]
                top, left, bottom, right = box
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))
                width = right-left
                height = bottom-top
                box_tmp.append(left)
                box_tmp.append(top)
                box_tmp.append(width)
                box_tmp.append(height)
                boxes.append(box_tmp)
            return boxes, score, predicted_class
        else:
            if count:
                print("top_label:", top_label)
                classes_nums = np.zeros([self.num_classes])
                for i in range(self.num_classes):
                    num = np.sum(top_label == i)
                    if num > 0:
                        print(self.class_names[i], " : ", num)
                    classes_nums[i] = num
                print("classes_nums:", classes_nums)
            if crop:
                for i, c in list(enumerate(top_boxes)):
                    top, left, bottom, right = top_boxes[i]
                    top = max(0, np.floor(top).astype('int32'))
                    left = max(0, np.floor(left).astype('int32'))
                    bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                    right = min(image.size[0], np.floor(right).astype('int32'))
                    dir_save_path = "img_crop"
                    if not os.path.exists(dir_save_path):
                        os.makedirs(dir_save_path)
                    crop_image = image.crop([left, top, right, bottom])
                    crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                    print("save crop_" + str(i) + ".png to " + dir_save_path)
            for i, c in list(enumerate(top_label)):
                predicted_class = self.class_names[int(c)]
                box = top_boxes[i]
                score = top_conf[i]
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
        mask = np.zeros((image.size[1], image.size[0]))
        for sub_output in output:
            b, h, w, c = np.shape(sub_output)
            sub_output = np.reshape(sub_output, [b, h, w, 3, -1])[0]
            score = np.max(sigmoid(sub_output[..., 4]), -1)
            score = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score = (score * 255).astype('uint8')
            mask = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()
        
    def close_session(self):
        self.sess.close()

def Inference_YOLOV7Model(config, model_path = './model/village_Detection_yolov7_l_2022_10_28.pth'):
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
        tiny = config.tiny
    )
    return yolo
