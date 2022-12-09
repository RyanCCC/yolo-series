import datetime
import os
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow as tf
from functools import partial
from . import get_train_model, yolo_body, get_lr_scheduler, ModelCheckpoint, YoloDatasets, get_anchors, get_classes




def yolov5(config):
    os.environ['CUDA_VISIABLE_DEVICES'] = config.gpus
    classes_path = config.classes_path
    # yolov5_anchors.txt
    anchor_path = config.anchors_path
    anchor_mask = config.ANCHOR_MASK
    pre_train_model = config.pretrain_weight
    input_shape = config.input_shape
    # 数据增强
    mosaic = config.mosaic
    mosaic_prob = config.mosaic_prob
    if mosaic:
        # mixup数据增强
        mixup = config.mixup
        mixup_prob = config.mixup_prob  
    else:
        mixup = config.mixup
        mixup_prob = -1
    special_aug_ratio = config.special_aug_ratio
    label_smoothing = config.label_smoothing
    epoch = config.epoch
    batch_size = config.batch_size
    learning_rate = config.learning_rate_unfreeze
    min_learning_rate = learning_rate*0.01
    # 优化器
    optimizer_type = 'sgd'
    momentum = 0.937
    weight_decay = 5e-4
    # options: cos, step
    learning_rate_decay_type = 'cos'
    save_dir = config.logdir
    saved_weight_name = config.save_weight
    train_annotation_path = config.train_txt
    val_annotation_path = config.val_txt
    anchors_mask = config.ANCHOR_MASK
    class_names, num_classes = get_classes(classes_path)
    anchors= get_anchors(anchor_path)
    num_anchors = len(anchors)
    phi = config.phi

    # 冻结训练
    Freeze_Train = True
    Freeze_Epoch = 50
    UnFreeze_Epoch = 100


    # 创建yolo模型
    model_body  = yolo_body((None, None, 3), anchor_mask, num_classes, phi)
    if pre_train_model != '':
        print('Load weights {}.'.format(pre_train_model))
        model_body.load_weights(pre_train_model, by_name=True, skip_mismatch=True)
    model =  get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing)

    # 获取数据集
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step  = num_train // batch_size * epoch
    if total_step <= wanted_step:
        if num_train // batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // batch_size) + 1
        print("\n\033[1;33;44m[Warning] 使用%s优化器时，建议将训练总步长设置到%d以上。\033[0m"%(optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] 本次运行的总训练数据量为%d，Unfreeze_batch_size为%d，共训练%d个Epoch，计算出总训练步长为%d。\033[0m"%(num_train, batch_size, epoch, total_step))
        print("\033[1;33;44m[Warning] 由于总训练步长为%d，小于建议总步长%d，建议设置总世代为%d。\033[0m"%(total_step, wanted_step, wanted_epoch))
    
    # 数据集加载
    train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchor_mask, 0, epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    val_dataloader = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchor_mask, 0, epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)

                                        
    # 设置超参数
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * learning_rate, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * min_learning_rate, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    # 设置callbacks
    time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    logging = TensorBoard(log_dir)
    checkpoint = ModelCheckpoint(os.path.join(save_dir, saved_weight_name), 
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = 1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
    lr_scheduler_func = get_lr_scheduler(learning_rate_decay_type, Init_lr_fit, Min_lr_fit, epoch)
    lr_scheduler = LearningRateScheduler(lr_scheduler_func, verbose = 1)
    callbacks = [logging, early_stopping, checkpoint, lr_scheduler]

    epoch_step = num_train // batch_size
    epoch_step_val  = num_val // batch_size
    train_dataloader.batch_size = batch_size
    val_dataloader.batch_size = batch_size

    # 设置Freeze train
    if Freeze_Train:
        freeze_layers = {'s': 125, 'm': 179, 'l': 234, 'x': 290}[phi]
        for i in range(freeze_layers): model_body.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
        optimizer = {
            'adam'  : Adam(lr = learning_rate, beta_1 = momentum),
            'sgd'   : SGD(lr = learning_rate, momentum = momentum, nesterov=True)
            }[optimizer_type]
        model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        print('Freeze Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit(
            x = train_dataloader,
            steps_per_epoch = epoch_step,
            validation_data = val_dataloader,
            validation_steps = epoch_step_val,
            epochs = Freeze_Epoch,
            callbacks = callbacks
        )
    for i in range(freeze_layers): model_body.layers[i].trainable = True
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
    # 设置超参数
    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * learning_rate, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * min_learning_rate, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    lr_scheduler_func = get_lr_scheduler(learning_rate_decay_type, Init_lr_fit, Min_lr_fit, epoch)

    train_dataloader = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchor_mask, 0, epoch, \
                                            mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    val_dataloader = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchor_mask, 0, epoch, \
                                            mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
    optimizer = {
            'adam'  : Adam(lr = learning_rate, beta_1 = momentum),
            'sgd'   : SGD(lr = learning_rate, momentum = momentum, nesterov=True)
            }[optimizer_type]
    model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    

    print('Freeze Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit(
        x = train_dataloader,
        steps_per_epoch = epoch_step,
        validation_data = val_dataloader,
        validation_steps = epoch_step_val,
        epochs = UnFreeze_Epoch,
        callbacks = callbacks
    )
    




    








