
import math
from functools import partial
import tensorflow as tf
from tensorflow.keras import backend as K
from nets.loss import _smooth_labels
from .tools import get_anchors_and_decode
from nets.ious import box_ciou


def yolo_loss(
    args, 
    input_shape, 
    anchors, 
    anchors_mask, 
    num_classes, 
    balance         = [0.4, 1.0, 4], 
    label_smoothing = 0.01, 
    box_ratio       = 0.05, 
    obj_ratio       = 1, 
    cls_ratio       = 0.5
):
    num_layers = len(anchors_mask)

    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]

    # input_shpae：416,416 
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    loss    = 0
    for l in range(num_layers):
        object_mask = y_true[l][..., 4:5]
        true_class_probs = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
             anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)
        
        pred_box = K.concatenate([pred_xy, pred_wh])

        raw_true_box    = y_true[l][...,0:4]
        ciou            = box_ciou(pred_box, raw_true_box)
        ciou_loss       = object_mask * (1 - ciou)
        
       
        tobj            = tf.where(tf.equal(object_mask, 1), tf.maximum(ciou, tf.zeros_like(ciou)), tf.zeros_like(ciou))
        confidence_loss = K.binary_crossentropy(tobj, raw_pred[..., 4:5], from_logits=True)
        class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)

        location_loss   = K.sum(ciou_loss) * box_ratio / num_pos
        confidence_loss = K.mean(confidence_loss) * balance[l] * obj_ratio
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss    += location_loss + confidence_loss + class_loss
        # if print_loss:
        # loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss], message='loss: ')
        
    return loss

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.05, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.05, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return 