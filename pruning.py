import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from nets.loss import yolo_loss
from nets.yolo4 import yolo_body
from utils.utils import ModelCheckpoint
from customerConf import YOLOV4Config as sys_config
from utils.dataloader import data_generator, get_classes, get_anchors
import os
import tensorflow_model_optimization as tfmot

'''
模型量化
'''

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

Epoch = sys_config.epoch
train_txt = sys_config.train_txt
log_dir = sys_config.logdir
classes_path = sys_config.classes_path
anchors_path = sys_config.anchors_path
weights_path = sys_config.pretrain_weight
save_model_name = sys_config.save_model_name
input_shape = (sys_config.imagesize,sys_config.imagesize)
anchor_mask = sys_config.ANCHOR_MASK
eager = sys_config.eager
normalize = sys_config.normalize

class_names = get_classes(classes_path)
anchors     = get_anchors(anchors_path)
num_classes = len(class_names)
num_anchors = len(anchors)

mosaic = sys_config.mosaic
Cosine_scheduler = sys_config.Cosine_scheduler
label_smoothing = sys_config.label_smoothing
learning_rate_freeze  = sys_config.learning_rate_freeze
regularization = sys_config.regularization

image_input = Input(shape=(None, None, 3))
h, w = input_shape
model_body = yolo_body(image_input, num_anchors//3, num_classes, sys_config.ATTENTION)
print('加载权重')
model_body.load_weights('./model/village_tf2.h5', by_name=True, skip_mismatch=True)
    

# 将模型的输出作为loss
y_true = [Input(shape=(h//{0:32, 1:16, 2:8}[l], w//{0:32, 1:16, 2:8}[l], \
        num_anchors//3, num_classes+5)) for l in range(3)]
loss_input = [*model_body.output, *y_true]
model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5, 'label_smoothing': label_smoothing})(loss_input)

model = Model([model_body.input, *y_true], model_loss)


logging = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir+save_model_name, save_weights_only=True, save_best_only=True, period=1)
early_stopping = EarlyStopping(min_delta=0, patience=10, verbose=1)

val_split = 0.1
with open(train_txt) as f:
    lines = f.readlines()
np.random.seed(10101)
np.random.shuffle(lines)
np.random.seed(None)
num_val = int(len(lines)*val_split)
num_train = len(lines) - num_val
    
Init_epoch          = sys_config.Init_epoch
Freeze_epoch        = sys_config.Freeze_epoch
batch_size          = sys_config.batch_size
        
epoch_size      = num_train // batch_size
epoch_size_val  = num_val // batch_size

reduce_lr       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

if epoch_size == 0 or epoch_size_val == 0:
    raise ValueError("数据集过小，无法进行训练，请扩充数据集。")
# Compute end step to finish pruning after 2 epochs.
end_step = np.ceil(num_train/batch_size).astype(np.int32) * Epoch
# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.50,final_sparsity=0.80,begin_step=0,end_step=end_step)}
model_for_pruning = prune_low_magnitude(model, **pruning_params)

model_for_pruning.summary()
model_for_pruning.compile(optimizer=Adam(learning_rate_freeze), loss={'prune_low_magnitude_yolo_loss': lambda y_true, y_pred: y_pred})
model_for_pruning.fit(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic, random=True, eager=False),
                    steps_per_epoch=epoch_size,
                    validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False, random=False, eager=False),
                    validation_steps=epoch_size_val,
                    epochs=Epoch,
                    initial_epoch=0,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping, tfmot.sparsity.keras.UpdatePruningStep()])
exported_model = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
# 以h5格式保存模型
exported_model.save('./model/village_model_pruning', save_format='tf')