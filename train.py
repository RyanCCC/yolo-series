from functools import partial

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                        TensorBoard)
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from nets.loss import yolo_loss
from nets.yolo4 import yolo_body
from utils.utils import (ModelCheckpoint,
                         WarmUpCosineDecayScheduler)
import config as sys_config
from utils.dataloader import data_generator, get_classes, get_anchors


# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(imgs, yolo_loss, targets, net, optimizer, regularization, normalize):
        with tf.GradientTape() as tape:
            # 计算loss
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args = [P5_output, P4_output, P3_output] + targets
            loss_value = yolo_loss(args,anchors,num_classes,label_smoothing=label_smoothing,normalize=normalize)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def fit_one_epoch(net, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, anchors, 
                        num_classes, label_smoothing, regularization=False, train_step=None):
    loss = 0
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets = [target0, target1, target2]
            targets = [tf.convert_to_tensor(target) for target in targets]
            loss_value = train_step(images, yolo_loss, targets, net, optimizer, regularization, normalize)
            loss = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer._decayed_lr(tf.float32).numpy()})
            pbar.update(1)
            
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets = [target0, target1, target2]
            targets = [tf.convert_to_tensor(target) for target in targets]

            P5_output, P4_output, P3_output = net(images)
            args = [P5_output, P4_output, P3_output] + targets
            loss_value = yolo_loss(args,anchors,num_classes,label_smoothing=label_smoothing, normalize=normalize)
            if regularization:
                # 加入正则化损失
                loss_value = tf.reduce_sum(net.losses) + loss_value
            # 更新验证集loss
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss)/ (iteration + 1)})
            pbar.update(1)

    logs = {'loss': loss.numpy()/(epoch_size+1), 'val_loss': val_loss.numpy()/(epoch_size_val+1)}
    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),loss/(epoch_size+1),val_loss/(epoch_size_val+1)))

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if __name__ == "__main__":
    annotation_path = sys_config.annotation_path
    log_dir = sys_config.logdir
    classes_path = sys_config.classes_path
    anchors_path = sys_config.anchors_path
    weights_path = sys_config.pretrain_weight
    save_model_name = sys_config.save_model_name
    input_shape = (sys_config.imagesize,sys_config.imagesize)
    eager = sys_config.eager
    normalize = sys_config.normalize

    class_names = get_classes(classes_path)
    anchors     = get_anchors(anchors_path)
    num_classes = len(class_names)
    num_anchors = len(anchors)

    mosaic = sys_config.mosaic
    Cosine_scheduler = sys_config.Cosine_scheduler
    label_smoothing = sys_config.label_smoothing

    regularization = sys_config.regularization

    image_input = Input(shape=(None, None, 3))
    h, w = input_shape
    print('Create YOLOv4 model with {} anchors and {} classes.'.format(num_anchors, num_classes))
    model_body = yolo_body(image_input, num_anchors//3, num_classes, sys_config.ATTENTION)
    print('Load weights {}.'.format(weights_path))
    model_body.load_weights(weights_path, by_name=True, skip_mismatch=True)
    
    #------------------------------------------------------#
    #   在这个地方设置损失，将网络的输出结果传入loss函数
    #   把整个模型的输出作为loss
    #------------------------------------------------------#
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
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines)*val_split)
    num_train = len(lines) - num_val
    
    freeze_layers = sys_config.freeze_layers
    for i in range(freeze_layers): model_body.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))

    if True:
        Init_epoch          = sys_config.Init_epoch
        Freeze_epoch        = sys_config.Freeze_epoch
        batch_size          = sys_config.batch_size
        learning_rate_freeze  = sys_config.learning_rate_freeze
        
        epoch_size      = num_train // batch_size
        epoch_size_val  = num_val // batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        if eager:
            gen     = tf.data.Dataset.from_generator(partial(data_generator, annotation_lines = lines[:num_train], batch_size = batch_size,
                input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=mosaic, random=True), (tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(data_generator, annotation_lines = lines[num_train:], batch_size = batch_size, 
                input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=False, random=False), (tf.float32, tf.float32, tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
            gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

            if Cosine_scheduler:
                lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate = learning_rate_freeze, first_decay_steps = 5 * epoch_size, t_mul = 1.0, alpha = 1e-2)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate_freeze, decay_steps=epoch_size, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            if Cosine_scheduler:
                # 预热期
                warmup_epoch    = int((Freeze_epoch-Init_epoch)*0.2)
                # 总共的步长
                total_steps     = int((Freeze_epoch-Init_epoch) * num_train / batch_size)
                # 预热步长
                warmup_steps    = int(warmup_epoch * num_train / batch_size)
                # 学习率
                reduce_lr       = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_freeze, total_steps=total_steps,
                                                            warmup_learning_rate=1e-4, warmup_steps=warmup_steps,
                                                            hold_base_rate_steps=num_train, min_learn_rate=1e-6)
                model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            else:
                reduce_lr       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
                model.compile(optimizer=Adam(learning_rate_freeze), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            for epoch in range(Init_epoch,Freeze_epoch):
                fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val,gen, gen_val, 
                            Freeze_epoch, anchors, num_classes, label_smoothing, regularization, get_train_step_fn())
        else:
            model.fit(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic, random=True, eager=False),
                    steps_per_epoch=epoch_size,
                    validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False, random=False, eager=False),
                    validation_steps=epoch_size_val,
                    epochs=Freeze_epoch,
                    initial_epoch=Init_epoch,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])

    for i in range(freeze_layers): model_body.layers[i].trainable = True

    # Transfer Learning
    if True:
        Freeze_epoch        = sys_config.Freeze_epoch
        Epoch               = sys_config.epoch
        batch_size          = sys_config.batch_size
        learning_rate_unfreeze  = sys_config.learning_rate_unfreeze

        epoch_size      = num_train // batch_size
        epoch_size_val  = num_val // batch_size

        if epoch_size == 0 or epoch_size_val == 0:
            raise ValueError("数据集过小，无法进行训练，请扩充数据集。")

        if eager:
            gen     = tf.data.Dataset.from_generator(partial(data_generator, annotation_lines = lines[:num_train], batch_size = batch_size,
                input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=mosaic, random=True), (tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(data_generator, annotation_lines = lines[num_train:], batch_size = batch_size, 
                input_shape = input_shape, anchors = anchors, num_classes = num_classes, mosaic=False, random=False), (tf.float32, tf.float32, tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)
            gen_val = gen_val.shuffle(buffer_size=batch_size).prefetch(buffer_size=batch_size)

            if Cosine_scheduler:
                lr_schedule = tf.keras.experimental.CosineDecayRestarts(
                    initial_learning_rate = learning_rate_unfreeze, first_decay_steps = 5 * epoch_size, t_mul = 1.0, alpha = 1e-2)
            else:
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=learning_rate_unfreeze, decay_steps=epoch_size, decay_rate=0.92, staircase=True)
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        else:
            if Cosine_scheduler:
                # 预热期
                warmup_epoch    = int((Epoch-Freeze_epoch)*0.2)
                # 总共的步长
                total_steps     = int((Epoch-Freeze_epoch) * num_train / batch_size)
                # 预热步长
                warmup_steps    = int(warmup_epoch * num_train / batch_size)
                # 学习率
                reduce_lr       = WarmUpCosineDecayScheduler(learning_rate_base=learning_rate_unfreeze, total_steps=total_steps,
                                                            warmup_learning_rate=1e-4, warmup_steps=warmup_steps,
                                                            hold_base_rate_steps=num_train, min_learn_rate=1e-6)
                model.compile(optimizer=Adam(), loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            else:
                reduce_lr       = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
                model.compile(optimizer=Adam(learning_rate_unfreeze), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        if eager:
            for epoch in range(Freeze_epoch,Epoch):
                fit_one_epoch(model_body, yolo_loss, optimizer, epoch, epoch_size, epoch_size_val,gen, gen_val, 
                            Epoch, anchors, num_classes, label_smoothing, regularization, get_train_step_fn())
        else:
            model.fit(data_generator(lines[:num_train], batch_size, input_shape, anchors, num_classes, mosaic=mosaic, random=True, eager=False),
                    steps_per_epoch=epoch_size,
                    validation_data=data_generator(lines[num_train:], batch_size, input_shape, anchors, num_classes, mosaic=False, random=False, eager=False),
                    validation_steps=epoch_size_val,
                    epochs=Epoch,
                    initial_epoch=Freeze_epoch,
                    callbacks=[logging, checkpoint, reduce_lr, early_stopping])
