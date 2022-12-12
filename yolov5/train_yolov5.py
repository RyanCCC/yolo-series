import datetime
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .nets.yolov5 import YoloBody
from .lib.loss import ModelEMA, YOLOLoss, get_lr_scheduler, set_optimizer_lr, weights_init
from .lib.callbacks import LossHistory, EvalCallback, EarlyStopping
from .lib.dataloader import YoloDataset, yolo_dataset_collate
from .lib.tools import download_weights, get_anchors, get_classes, show_config
from .lib.tools_fit import fit_one_epoch


def yolov5(config):
    # GPU SETTING
    Cuda = config.cuda
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus
    # 是否使用单机多卡分布式运行
    distributed = False
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    sync_bn = False
    # 混合精度模式
    fp16 = False
    classes_path =  config.classes_path
    anchors_path = config.anchors_path
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    model_path = config.pretrain_weight
    input_shape = config.input_shape
    backbone = config.backbone
    # 是否加载backbone权重，优先级比model_path要低
    pretrained = False
    #   phi：n、s、m、l、x
    phi = config.phi
    mosaic = config.mosaic
    mosaic_prob = config.mosaic_prob
    mixup = config.mixup
    mixup_prob = config.mixup_prob 
    special_aug_ratio = config.special_aug_ratio
    # 标签平滑
    label_smoothing = config.label_smoothing

    Freeze_Train = config.Freeze_Train
    Init_Epoch = config.Init_epoch
    Freeze_Epoch = config.Freeze_epoch
    Freeze_batch_size = config.batch_size
    UnFreeze_Epoch = config.epoch
    Unfreeze_batch_size = config.Unfreeze_batch_size
    
    Init_lr = config.learning_rate_unfreeze
    Min_lr = Init_lr * 0.01

    optimizer_type = "sgd"
    momentum = 0.937
    weight_decay = 5e-4

    lr_decay_type = "cos"
    # 权重保存参数设置
    save_period = 10
    save_dir = config.logdir
    # 训练时推理的参数
    eval_flag = config.eval_flag
    eval_period = 10
    num_workers = 2

    # 数据集参数
    train_annotation_path = config.train_txt
    val_annotation_path = config.val_txt

    # 设置显卡
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0


    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)


    if pretrained:
        if distributed:
            if local_rank == 0:
                download_weights(phi, model_dir='./yolov5/checkpoints')  
            dist.barrier()
        else:
            download_weights(phi, model_dir='./yolov5/checkpoints')

    # 初始化模型
    model = YoloBody(anchors_mask, num_classes, phi, backbone, pretrained=pretrained, input_shape=input_shape)
    if not pretrained:
        weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # display key
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))



    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask, label_smoothing)

    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None
        
    # 仅支持torch1.6以上
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()
    # 分布式训练
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()
            
    # 权值平滑
    ema = ModelEMA(model_train)
    
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            classes_path = classes_path, anchors_path = anchors_path, anchors_mask = anchors_mask, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
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

    UnFreeze_flag = False
    if Freeze_Train:
        for param in model.backbone.parameters():
            param.requires_grad = False

    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

    nbs = 64
    lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
    lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
    Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
    Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

    pg0, pg1, pg2 = [], [], []  
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)    
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)   
    optimizer = {
        'adam'  : optim.Adam(pg0, Init_lr_fit, betas = (momentum, 0.999)),
        'sgd'   : optim.SGD(pg0, Init_lr_fit, momentum = momentum, nesterov=True)
    }[optimizer_type]
    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2})

    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
    epoch_step      = num_train // batch_size
    epoch_step_val  = num_val // batch_size
        
    if epoch_step == 0 or epoch_step_val == 0:
        raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    if ema:
        ema.updates     = epoch_step * Init_Epoch
        
    train_dataset   = YoloDataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, \
                                        mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True, special_aug_ratio=special_aug_ratio)
    val_dataset     = YoloDataset(val_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, \
                                        mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False, special_aug_ratio=0)
        
    if distributed:
        train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
        batch_size = batch_size // ngpus_per_node
        shuffle = False
    else:
        train_sampler   = None
        val_sampler     = None
        shuffle         = True

    train_data = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
    val_data = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

    if local_rank == 0:
        eval_callback   = EvalCallback(model, input_shape, anchors, anchors_mask, class_names, num_classes, val_lines, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
    else:
        eval_callback   = None

    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            batch_size = Unfreeze_batch_size

            nbs = 64
            lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
            lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
            Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
            Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

            lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

            for param in model.backbone.parameters():
                param.requires_grad = True

            epoch_step      = num_train // batch_size
            epoch_step_val  = num_val // batch_size

            if epoch_step == 0 or epoch_step_val == 0:
                raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
                    
            if ema:
                ema.updates     = epoch_step * epoch

            if distributed:
                batch_size  = batch_size // ngpus_per_node
                    
            train_data = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
            val_data = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

            UnFreeze_flag = True

        train_data.dataset.epoch_now = epoch
        val_data.dataset.epoch_now = epoch

        if distributed:
            train_sampler.set_epoch(epoch)

        set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

        fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, train_data, val_data, UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank)
            
        if distributed:
            dist.barrier()

    if local_rank == 0:
        loss_history.writer.close()
    




    








