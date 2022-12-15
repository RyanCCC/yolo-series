import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np


def sigmoid(x):  
    return np.exp(-np.logaddexp(0, -x))

def non_max_suppression(boxes, scores, threshold):	
    assert boxes.shape[0] == scores.shape[0]
    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]
    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]
    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []
    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                           areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]
    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])
    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])
    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])
    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])
    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)
    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections
    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious

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
    # outputs = outputs[:-1]
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


def yolo_correct_boxes_numpy(box_xy, box_wh, image_shape, input_shape, letterbox_image=True):
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = np.array(input_shape, np.float64)
    image_shape = np.array(image_shape, np.float64)

    if letterbox_image:
        new_shape = np.round(image_shape * np.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
    boxes *= np.concatenate([image_shape, image_shape], axis=1)
    return boxes

def DecodeBox_numpy(outputs,image_shape, input_shape, class_names,confidence=0.5, max_boxes=100, letterbox_image = True):
    num_classes = len(class_names)
    batch_size = np.shape(outputs[0])[0]
    grids = []
    strides = []
    hw = [np.shape(x)[1:3] for x in outputs]
    '''
    outputs before:
    batch_size, 80, 80, 4+1+num_classes
    batch_size, 40, 40, 4+1+num_classes
    batch_size, 20, 20, 4+1+num_classes

    outputs after:
    batch_size, 8400, 8400, 4+1+num_classes
    '''
    outputs = np.concatenate([np.reshape(x, [batch_size, -1, 5 + num_classes]) for x in outputs], axis = 1)
    for i in range(len(hw)):
        grid_x, grid_y  = np.meshgrid(np.arange(hw[i][1]), np.arange(hw[i][0]))
        grid = np.reshape(np.stack((grid_x, grid_y), 2), (1, -1, 2))
        shape  = np.shape(grid)[:2]
        grids.append(np.array(grid, np.float32))
        strides.append(np.ones((shape[0], shape[1], 1)) * input_shape[0] / np.array(hw[i][0], np.float32))
    grids = np.concatenate(grids, axis=1)
    strides = np.concatenate(strides, axis=1)
    box_xy = (outputs[..., :2] + grids) * strides / np.array(input_shape[::-1], np.float32)
    box_wh = np.exp(outputs[..., 2:4]) * strides / np.array(input_shape[::-1], np.float32)
    box_confidence  = sigmoid(outputs[..., 4:5])
    box_class_probs = sigmoid(outputs[..., 5: ])
    boxes = yolo_correct_boxes_numpy(box_xy, box_wh, image_shape, input_shape)
    box_scores  = box_confidence * box_class_probs

    mask = box_scores >= confidence
    max_boxes_tensor = max_boxes
    boxes_out   = np.empty(shape=[0, 4])
    scores_out  = np.array([])
    classes_out = np.array([], dtype=np.int32)
    for c in range(num_classes):
        class_boxes =np.array(boxes[mask[..., c]])
        class_box_scores = np.array(box_scores[..., c][mask[..., c]])
        # nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=0.5)
        # nms_index = nms_index.numpy
        nms_index = non_max_suppression(class_boxes, class_box_scores, threshold=0.5)

        if len(class_boxes) == 0:
            continue
        class_boxes = class_boxes[nms_index]
        class_box_scores = class_box_scores[nms_index]
        classes = np.ones_like(class_box_scores, 'int32') * c

        boxes_out = np.append(boxes_out, class_boxes, axis=0)
        scores_out = np.append(scores_out, class_box_scores)
        classes_out = np.append(classes_out, classes)


    return boxes_out, scores_out, classes_out