from utils.utils import get_random_data_with_Mosaic, get_random_data
import numpy as np
import tensorflow as tf
import config

def data_generator(annotation_lines, batch_size, input_shape, anchors, num_classes, mosaic=False, random=True, eager=True):
    n = len(annotation_lines)
    i = 0
    flag = True
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i==0:
                # 随机打乱训练集
                np.random.shuffle(annotation_lines)
            if mosaic:
                if flag and (i+4) < n:
                    image, box = get_random_data_with_Mosaic(annotation_lines[i:i+4], input_shape)
                    i = (i+4) % n
                else:
                    image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                    i = (i+1) % n
                flag = bool(1-flag)
            else:
                image, box = get_random_data(annotation_lines[i], input_shape, random=random)
                i = (i+1) % n
            image_data.append(image)
            box_data.append(box)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        y_true = preprocess_true_boxes(box_data, input_shape, anchors, num_classes)
        if eager:
            yield image_data, y_true[0], y_true[1], y_true[2]
        else:
            yield [image_data, *y_true], np.zeros(batch_size)

#  详解：https://blog.csdn.net/weixin_38145317/article/details/95349201
# https://zhuanlan.zhihu.com/p/79425557
'''
preprocess_true_boxes，顾名思义对真实框进行相关预处理。
为了更好地理解这个函数，有必要先说明bboxes和label分别表示什么，具体怎么操作
bboxes是用来存放真实框的中心坐标以及宽高(x,y,w,h),其shape为(3,150,4)。3表示3种网格尺寸，150表示每种网格尺寸允许存放的最大真实框数量，4就是(x,y,w,h)。
label是用来存放3种网格尺寸下每一个网格的中心坐标、宽高、置信度以及所属类别(x, y, w, h, conf, classid)，其中class_id用one-hot编码表示，并对其进行平滑处理。
label的shape为(3,train_output_sizes,train_output_sizes,anchor_per_scale,5 + num_classes)。3表示3种网格尺寸，train_output_sizes表示每种网格尺寸的大小，anchor_per_scale表示每个网格预测多少个anchor框，5 + numclasses就不再多说了。
label的初始化为0矩阵，即每个网格的信息(x, y, w, h, conf, classid)都设置为0。计算3个先验框和真实框的iou值，筛选iou值>0.3的先验框并标记索引， 然后将真实框的(x,y,w,h,class_id)填充到真实框所属的网格中(对应标记索引)，网格的置信度设为1。
'''
def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    # 该函数得到detectors_mask（最佳预测的anchor boxes，每一个true boxes都对应一个anchor boxes）
    # true_boxes：实际框的位置和类别
    assert (true_boxes[..., 4]<num_classes).all()
    num_layers = len(anchors)//3
    anchor_mask = config.ANCHOR_MASK

    # 归一化
    true_boxes = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    # 中心点坐标
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    # 标注框长宽
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    # {0:32  1:16  2:8}表示特征金字塔下采样的倍数，得到不同尺度下的feature map的尺寸
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    # 对不同的特征图下生成对应目标框的，如目标框有两个，对于13*13feature map，生成2*13*13*3*(5+class_num+1)
    y_true = [np.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    anchors = np.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    # 对每一张图进行处理，计算每个目标与设定anchor box的最佳IOU
    for b in range(m):
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        wh = np.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        intersect_mins = np.maximum(box_mins, anchor_mins)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = np.argmax(iou, axis=-1)

        for t, best_detect in enumerate(best_anchor):
            for l in range(num_layers):
                if best_detect in anchor_mask[l]:
                    # 根据真实框的坐标信息来计算所属网格左上角的位置. xind, yind其实就是网格的坐标
                    xind = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    yind = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(best_detect)
                    c = true_boxes[b, t, 4].astype('int32')
                    # 填充真实框的中心位置和宽高
                    y_true[l][b, yind, xind, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, yind, xind, k, 4] = 1
                    y_true[l][b, yind, xind, k, 5+c] = 1

    return y_true


# tensorflow的numpy
def preprocess_true_boxes_tf(true_boxes, input_shape, anchors, num_classes):
    # assert (true_boxes[..., 4]<num_classes).numpy().all()
    num_layers = len(anchors)//3
    anchor_mask = config.ANCHOR_MASK

    true_boxes = tf.cast(true_boxes, tf.float32)
    input_shape = tf.cast(input_shape, tf.float32)
    boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
    boxes_wh = true_boxes[..., 2:4] - true_boxes[..., 0:2]
    true_boxes[..., 0:2] = boxes_xy/input_shape[::-1]
    true_boxes[..., 2:4] = boxes_wh/input_shape[::-1]

    # m为图片数量，grid_shapes为网格的shape
    m = true_boxes.shape[0]
    grid_shapes = [input_shape//{0:32, 1:16, 2:8}[l] for l in range(num_layers)]
    y_true = [tf.zeros((m,grid_shapes[l][0],grid_shapes[l][1],len(anchor_mask[l]),5+num_classes),
        dtype='float32') for l in range(num_layers)]
    anchors = tf.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = boxes_wh[..., 0]>0

    for b in range(m):
        # 对每一张图进行处理
        wh = boxes_wh[b, valid_mask[b]]
        if len(wh)==0: continue
        wh = tf.expand_dims(wh, -2)
        box_maxes = wh / 2.
        box_mins = -box_maxes
        intersect_mins = tf.maximum(box_mins, anchor_mins)
        intersect_maxes = tf.minimum(box_maxes, anchor_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]

        box_area = wh[..., 0] * wh[..., 1]
        anchor_area = anchors[..., 0] * anchors[..., 1]

        iou = intersect_area / (box_area + anchor_area - intersect_area)
        best_anchor = tf.argmax(iou, axis=-1)

        for t, n in enumerate(best_anchor):
            for l in range(num_layers):
                if n in anchor_mask[l]:
                    i = tf.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                    j = tf.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                    k = anchor_mask[l].index(n)
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 5+c] = 1

    return y_true



def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


@tf.function
def transform_targets_for_output(y_true, grid_size, anchor_idxs):
    # y_true: (N, boxes, (x1, y1, x2, y2, class, best_anchor))
    N = tf.shape(y_true)[0]

    # y_true_out: (N, grid, grid, anchors, [x1, y1, x2, y2, obj, class])
    y_true_out = tf.zeros(
        (N, grid_size, grid_size, tf.shape(anchor_idxs)[0], 6))

    anchor_idxs = tf.cast(anchor_idxs, tf.int32)

    indexes = tf.TensorArray(tf.int32, 1, dynamic_size=True)
    updates = tf.TensorArray(tf.float32, 1, dynamic_size=True)
    idx = 0
    for i in tf.range(N):
        for j in tf.range(tf.shape(y_true)[1]):
            if tf.equal(y_true[i][j][2], 0):
                continue
            anchor_eq = tf.equal(
                anchor_idxs, tf.cast(y_true[i][j][5], tf.int32))

            if tf.reduce_any(anchor_eq):
                box = y_true[i][j][0:4]
                box_xy = (y_true[i][j][0:2] + y_true[i][j][2:4]) / 2

                anchor_idx = tf.cast(tf.where(anchor_eq), tf.int32)
                # grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32)
                grid_xy = tf.cast(box_xy // (1/grid_size), tf.int32) 
                for element in grid_xy: 
                    if element >= grid_size: 
                        grid_xy = grid_xy // 2 
                        box = box / 2

                # grid[y][x][anchor] = (tx, ty, bw, bh, obj, class)
                indexes = indexes.write(
                    idx, [i, grid_xy[1], grid_xy[0], anchor_idx[0][0]])
                updates = updates.write(
                    idx, [box[0], box[1], box[2], box[3], 1, y_true[i][j][4]])
                idx += 1

    # tf.print(indexes.stack())
    # tf.print(updates.stack())

    return tf.tensor_scatter_nd_update(
        y_true_out, indexes.stack(), updates.stack())


def transform_targets(y_train, anchors, anchor_masks, size):
    y_outs = []
    grid_size = size // 32

    # calculate anchor index for true boxes
    anchors = tf.cast(anchors, tf.float32)
    anchor_area = anchors[..., 0] * anchors[..., 1]
    box_wh = y_train[..., 2:4] - y_train[..., 0:2]
    box_wh = tf.tile(tf.expand_dims(box_wh, -2),
                     (1, 1, tf.shape(anchors)[0], 1))
    box_area = box_wh[..., 0] * box_wh[..., 1]
    intersection = tf.minimum(box_wh[..., 0], anchors[..., 0]) * \
        tf.minimum(box_wh[..., 1], anchors[..., 1])
    iou = intersection / (box_area + anchor_area - intersection)
    anchor_idx = tf.cast(tf.argmax(iou, axis=-1), tf.float32)
    anchor_idx = tf.expand_dims(anchor_idx, axis=-1)

    y_train = tf.concat([y_train, anchor_idx], axis=-1)

    for anchor_idxs in anchor_masks:
        y_outs.append(transform_targets_for_output(
            y_train, grid_size, anchor_idxs))
        grid_size *= 2

    return tuple(y_outs)


if __name__ == '__main__':
    '''
    debug preprocess_true_boxes
    '''
    true_boxes = [[[263, 211, 324, 339, 8], [165, 264, 253, 372, 8], [241, 194, 295, 299, 8], [150, 141, 229, 284, 14]],
              [[69, 172, 270, 330, 12], [150, 141, 229, 284, 14], [241, 194, 295, 299, 8], [285, 201, 327, 331, 14]]]

    true_boxes = np.array(true_boxes)
    print(true_boxes.shape)
    input_shape = (416, 416)
    anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    num_classes = 20
    
    preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes)