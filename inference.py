import tensorflow as tf
from tensorflow.keras import backend as K
from PIL import Image,ImageDraw, ImageFont
from utils.utils import letterbox_image
import config
import numpy as np
import os
import colorsys
import tensorflow_model_optimization as tfmot

# 加载模型
model = tf.keras.models.load_model(model_path)

@tf.function
def get_outputs(model, image_data):
    outputs = model([image_data], training=False)
    return outputs

def get_class(classes_path):
        classes_path = os.path.expanduser(classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

def get_anchors(anchors_path):
        anchors_path = os.path.expanduser(anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


def get_colors(class_names):
    # 画框设置不同的颜色
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    # 打乱颜色
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)
    return colors

def yolo_boxes_and_scores(feats, anchors, num_classes, input_shape, image_shape, letterbox_image):
    box_xy, box_wh, box_confidence, box_class_probs = yolo_head(feats, anchors, num_classes, input_shape)
    
    if letterbox_image:
        boxes = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape)
    else:
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        box_mins = box_yx - (box_hw / 2.)
        box_maxes = box_yx + (box_hw / 2.)

        input_shape = K.cast(input_shape, K.dtype(box_yx))
        image_shape = K.cast(image_shape, K.dtype(box_yx))

        boxes =  K.concatenate([
            box_mins[..., 0:1] * image_shape[0],  # y_min
            box_mins[..., 1:2] * image_shape[1],  # x_min
            box_maxes[..., 0:1] * image_shape[0],  # y_max
            box_maxes[..., 1:2] * image_shape[1]  # x_max
        ])
    
    boxes = K.reshape(boxes, [-1, 4])
    box_scores = box_confidence * box_class_probs
    box_scores = K.reshape(box_scores, [-1, num_classes])
    return boxes, box_scores

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    new_shape = K.round(image_shape * K.min(input_shape/image_shape))
    offset = (input_shape - new_shape) / 2. / input_shape
    scale = input_shape / new_shape

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes = K.concatenate([
        box_mins[..., 0:1],  # y_min
        box_mins[..., 1:2],  # x_min
        box_maxes[..., 0:1],  # y_max
        box_maxes[..., 1:2]  # x_max
    ])

    boxes *= K.concatenate([image_shape, image_shape])
    return boxes


def yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #---------------------------------------------------#
    #   [1, 1, 1, num_anchors, 2]
    #---------------------------------------------------#
    feats = tf.convert_to_tensor(feats)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, 1, num_anchors, 2])

    
    grid_shape = K.shape(feats)[1:3]  # height, width
    grid_y = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]),
                    [1, grid_shape[1], 1, 1])
    grid_x = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]),
                    [grid_shape[0], 1, 1, 1])
    grid = K.concatenate([grid_x, grid_y])
    grid = K.cast(grid, K.dtype(feats))

    feats = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    # b_x = sigmoid(t_x)+C_x, b_y = sigmoid(t_y)+C_y, b_w = p_w*e^{t_x}，同理b_y
    box_xy = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[...,::-1], K.dtype(feats))
    box_wh = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[...,::-1], K.dtype(feats))
    box_confidence = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])

    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

def inference(model_path, image_path, letterbox_image=config.letterbox_image, score=config.score, iou=config.iou,max_boxes=config.max_boxes,
                class_path = config.classes_path, anchors=config.anchors_path):
    # 读取类别信息
    class_names = get_class(class_path)
    # 读取预设框信息
    anchors = get_anchors(anchors)
    # 设置颜色
    colors = get_colors(class_names)
    # 推理之前的图像预处理
    image = Image.open(image_path)
    image = image.convert('RGB')
    if letterbox_image:
        boxed_image = letterbox_image(image, (config.imagesize,config.imagesize))
    else:
        boxed_image = image.resize((config.imagesize,config.imagesize), Image.BICUBIC)
    image_data = np.array(boxed_image, dtype='float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    # input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
    image_shape = np.array([image.size[1], image.size[0]], dtype='float32')
    import time
    time1 = time.time()
    yolo_outputs = get_outputs(model, image_data)
    time2 = time.time()
    print(time2-time1)
    num_layers = len(yolo_outputs)
    anchor_mask = config.ANCHOR_MASK
    input_shape = K.shape(yolo_outputs[0])[1:3] * 32
    boxes = []
    box_scores = []

    for l in range(num_layers):
        _boxes, _box_scores = yolo_boxes_and_scores(yolo_outputs[l], anchors[anchor_mask[l]], len(class_names), input_shape,
                                                image_shape, letterbox_image)
        boxes.append(_boxes)
        box_scores.append(_box_scores)
    boxes = K.concatenate(boxes, axis=0)
    box_scores = K.concatenate(box_scores, axis=0)
    mask = box_scores >= score
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(len(class_names)):
        class_boxes = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
        # tensorflow非极大值抑制：https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
        nms_index = tf.image.non_max_suppression(
            class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=0.5)
        # K.gather 检索张量reference中索引indices的元素
        class_boxes = K.gather(class_boxes, nms_index)
        class_box_scores = K.gather(class_box_scores, nms_index)
        classes = K.ones_like(class_box_scores, 'int32') * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    out_boxes = K.concatenate(boxes_, axis=0)
    out_scores = K.concatenate(scores_, axis=0)
    out_classes = K.concatenate(classes_, axis=0)
        
    print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
    font = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = max((image.size[0] + image.size[1]) // 300, 1)
        
    for i, c in list(enumerate(out_classes)):
        predicted_class = class_names[c]
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
            draw.rectangle([left + i, top + i, right - i, bottom - i],outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)],fill=colors[c])
        draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
        del draw

    return image



def model_optimizer(model_path):
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    # 对模型中的卷积层进行剪枝
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            tfmot.sparsity.keras.prune_low_magnitude(layer)
    # pruning
    model = tfmot.sparsity.keras.strip_pruning(model)
    # 移动端需要转化为tflite
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # quantized_and_pruned_tflite_model = converter.convert()
    model.save('./tmp')


# TODO: FPS Test


# TODO：map Test




if __name__ == '__main__':
    model_path = './village_model/'
    # model_path = './tmp/'
    image_path = './result/20210817115925.jpg'
    # model_optimizer(model_path)
    r_image = inference(model_path=model_path, image_path=image_path)
    r_image.show()
