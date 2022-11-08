import datetime
import os
from functools import partial

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from .nets.yolov7 import get_train_model, yolo_body
from .nets.loss import get_lr_scheduler
from .lib.callbacks import ModelCheckpoint
from .lib.dataloader import YoloDatasets
from .lib.tools import get_anchors, get_classes, show_config
from tqdm import tqdm
from .nets.loss import yolo_loss

def get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, strategy):
    @tf.function
    def train_step(imgs, targets, net, optimizer):
        with tf.GradientTape() as tape:
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(
                args, input_shape, anchors, anchors_mask, num_classes, 
                balance=[0.4, 1.0, 4],
                box_ratio=0.05, 
                obj_ratio=1 * (input_shape[0] * input_shape[1]) / (640 ** 2),
                cls_ratio=0.5 * (num_classes / 80), 
                label_smoothing=label_smoothing
            )
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value

    if strategy == None:
        return train_step
    else:
        # multi gpu training
        @tf.function
        def distributed_train_step(images, targets, net, optimizer):
            per_replica_losses = strategy.run(train_step, args=(images, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_train_step

def get_val_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, strategy):
    @tf.function
    def val_step(imgs, targets, net, optimizer):
        P5_output, P4_output, P3_output = net(imgs, training=True)
        args        = [P5_output, P4_output, P3_output] + targets
        loss_value  = yolo_loss(
            args, input_shape, anchors, anchors_mask, num_classes, 
            balance=[0.4, 1.0, 4],
            box_ratio=0.05, 
            obj_ratio=1 * (input_shape[0] * input_shape[1]) / (640 ** 2),
            cls_ratio=0.5 * (num_classes / 80), 
            label_smoothing=label_smoothing
        )
        loss_value  = tf.reduce_sum(net.losses) + loss_value
        return loss_value
    if strategy == None:
        return val_step
    else:
        # multi gpu
        @tf.function
        def distributed_val_step(images, targets, net, optimizer):
            per_replica_losses = strategy.run(val_step, args=(images, targets, net, optimizer,))
            return strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses,
                                    axis=None)
        return distributed_val_step
                            
def fit_one_epoch(net, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, 
            input_shape, anchors, anchors_mask, num_classes, label_smoothing, save_period, save_dir, checkpoints, strategy):
    train_step  = get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, strategy)
    val_step = get_val_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing, strategy)
    
    loss = 0
    val_loss = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, target0, target1, target2, labels = batch[0], batch[1], batch[2], batch[3], batch[4]
            targets     = [target0, target1, target2, labels]
            loss_value  = train_step(images, targets, net, optimizer)
            loss        = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer.lr.numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, target0, target1, target2, labels = batch[0], batch[1], batch[2], batch[3], batch[4]
            targets = [target0, target1, target2, labels]
            loss_value = val_step(images, targets, net, optimizer)
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'val_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    eval_callback.on_epoch_end(epoch, logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    
    # save checkpoints
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, checkpoints))
        
    if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
        print('Save best model to best_epoch_weights.pth')
        net.save_weights(os.path.join(save_dir, "best_epoch_weights.h5"))



def yolov7(config):
    eager = False
    classes_path = config.classes_path
    anchors_path = config.anchor_path
    anchors_mask = config.ANCHOR_MASK
    model_path = config.pretrain_weight
    input_shape = config.input_shape
    phi =  config.phi
    mosaic = True
    mosaic_prob = 0.5
    mixup = True
    mixup_prob = 0.5
    special_aug_ratio = 0.7
    label_smoothing = config.label_smoothing
    Init_Epoch = config.Init_epoch
    Freeze_Epoch = config.Freeze_epoch
    Freeze_batch_size = config.batch_size

    UnFreeze_Epoch = config.epoch
    Unfreeze_batch_size = 4
    Freeze_Train = config.Freeze_Train
    
    Init_lr = config.learning_rate
    Min_lr = Init_lr * 0.01
    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4
    lr_decay_type = 'cos'
    save_dir = config.logdir
    train_annotation_path = config.train_txt
    val_annotation_path = config.val_txt
    os.environ["CUDA_VISIBLE_DEVICES"]  = config.gpus

    
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    # init model
    model_body  = yolo_body((None, None, 3), anchors_mask, num_classes, phi, weight_decay)
    if model_path != '':
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)
    if not eager:
        model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing)
            
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    show_config(
        classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, \
        Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
        Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
        save_dir = save_dir,  num_train = num_train, num_val = num_val
    )
    
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))


    if Freeze_Train:
        freeze_layers = {'n':118, 's': 118, 'm': 167, 'l': 216, 'x': 265}[phi]
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
            
        
    batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size

    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

    train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    val_dataloader = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, Init_Epoch, UnFreeze_Epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

    optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        
    start_epoch = Init_Epoch
    end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
    model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    logging = TensorBoard(log_dir)
    checkpoint      = ModelCheckpoint(os.path.join(save_dir, config.save_weight), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False)
    early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose = 1)
    callbacks = [logging, checkpoint, lr_scheduler, early_stopping]

    if start_epoch < end_epoch:
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(
            x = train_dataloader,
            steps_per_epoch = epoch_step,
            validation_data = val_dataloader,
            validation_steps = epoch_step_val,
            epochs = end_epoch,
            initial_epoch = start_epoch,
            callbacks = callbacks
        )
    if Freeze_Train:
        batch_size  = Unfreeze_batch_size
        start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
        end_epoch   = UnFreeze_Epoch
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks       = [logging, checkpoint, lr_scheduler]
                    
        for i in range(len(model_body.layers)): 
            model_body.layers[i].trainable = True
        model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        epoch_step = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        train_dataloader.batch_size    = Unfreeze_batch_size
        val_dataloader.batch_size      = Unfreeze_batch_size

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(
            x = train_dataloader,
            steps_per_epoch = epoch_step,
            validation_data = val_dataloader,
            validation_steps = epoch_step_val,
            epochs = end_epoch,
            initial_epoch = start_epoch,
            callbacks = callbacks
        )