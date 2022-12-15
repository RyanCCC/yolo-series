import tensorflow as tf
from tensorflow.keras import backend as K
from pathlib import Path

def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffixes
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            assert Path(f).suffix.lower() in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    grid_shape = K.shape(feats)[1:3]
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), feats.dtype)
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    box_xy          = (K.sigmoid(feats[..., :2]) * 2 - 0.5 + grid) / K.cast(grid_shape[..., ::-1], feats.dtype)
    box_wh          = (K.sigmoid(feats[..., 2:4]) * 2) ** 2 * anchors_tensor / K.cast(input_shape[::-1], feats.dtype)
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

# 后处理
def DecodeBox(outputs,
            anchors,
            num_classes,
            image_shape,
            input_shape,
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes = 100,
            confidence = 0.5,
            nms_iou = 0.3,
            letterbox_image = True):
    
    # image_shape = K.reshape(outputs[-1],[-1])

    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    for i in range(len(anchor_mask)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    box_xy          = K.concatenate(box_xy, axis = 0)
    box_wh          = K.concatenate(box_wh, axis = 0)
    box_confidence  = K.concatenate(box_confidence, axis = 0)
    box_class_probs = K.concatenate(box_class_probs, axis = 0)
    
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    box_scores  = box_confidence * box_class_probs
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out = K.concatenate(boxes_out, axis=0)
    scores_out = K.concatenate(scores_out, axis=0)
    classes_out = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out