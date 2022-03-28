import numpy as np
import tensorflow as tf
import config
from PIL import Image
import cv2
import math
from tensorflow import keras
from random import sample, shuffle


def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

def preprocess_input(image):
    image /= 255.0
    image -= np.array([0.485, 0.456, 0.406])
    image /= np.array([0.229, 0.224, 0.225])
    return image

class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, epoch_now, epoch_length, mosaic, train, mosaic_ratio = 0.7):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.epoch_now          = epoch_now - 1
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.train              = train

        self.mosaic_ratio       = mosaic_ratio

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data  = []
        box_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i = i % self.length
            if self.mosaic:
                if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                    lines = sample(self.annotation_lines, 3)
                    lines.append(self.annotation_lines[i])
                    shuffle(lines)
                    image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                else:
                    image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            else:
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
                
            if len(box) != 0:
                box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
                
            image_data.append(preprocess_input(np.array(image, np.float32)))
            box_data.append(box)

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)
        return [image_data, box_data], np.zeros(self.batch_size) 

    def generate(self):
        i = 0
        while True:
            image_data  = []
            box_data    = []
            for b in range(self.batch_size):
                if i==0:
                    np.random.shuffle(self.annotation_lines)
                if self.mosaic:
                    if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                        lines = sample(self.annotation_lines, 3)
                        lines.append(self.annotation_lines[i])
                        shuffle(lines)
                        image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                    else:
                        image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
                else:
                    image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)

                if len(box) != 0:
                    box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
                    box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
                    
                i = (i+1) % self.length
                image_data.append(preprocess_input(np.array(image, np.float32)))
                box_data.append(box)

            image_data  = np.array(image_data)
            box_data    = np.array(box_data)
            yield image_data, box_data

    def on_epoch_end(self):
        self.epoch_now += 1
        shuffle(self.annotation_lines)
        
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=500, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        image = Image.open(line[0])
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape
        box = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2
            image = image.resize((nw,nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0]  = 0
                box[:, 2][box[:, 2]>w]      = w
                box[:, 3][box[:, 3]>h]      = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                if len(box)>max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=500, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            line_content = line.split()
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            iw, ih = image.size
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names