import os
from functools import partial
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam

from yolox.nets.yolox import yolo_body, get_yolox_model
from yolox.lib.loss_yolox import get_lr_scheduler, get_yolo_loss
from utils.callbacks import ModelCheckpoint
from yolox.lib.dataloader import YoloDatasets, get_classes
from tqdm import tqdm


def get_train_step_fn():
    @tf.function
    def train_step(imgs, targets, net, yolo_loss, optimizer):
        with tf.GradientTape() as tape:
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + [targets]
            
            loss_value  = yolo_loss(args)
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step

def val_step(imgs, targets, net, yolo_loss, optimizer):
    P5_output, P4_output, P3_output = net(imgs, training=False)
    args        = [P5_output, P4_output, P3_output] + [targets]
    loss_value  = yolo_loss(args)
    loss_value  = tf.reduce_sum(net.losses) + loss_value
    return loss_value

def fit_one_epoch(net, yolo_loss, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, 
            input_shape, num_classes, save_period, save_dir):
    train_step  = get_train_step_fn()
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, targets = batch[0], batch[1]
            targets     = tf.convert_to_tensor(targets)
            loss_value  = train_step(images, targets, net, yolo_loss, optimizer)
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
            images, targets = batch[0], batch[1]
            targets     = tf.convert_to_tensor(targets)
            loss_value  = val_step(images, targets, net, yolo_loss, optimizer)
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights(os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.h5" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"]  = config.gpus
    classes_path = config.classes_path
    pretrain_model_path = config.pretrain_weight
    input_shape = [640,640]
    # yolox的版本：tiny、s、m、l、x
    phi = config.phi
    mosaic = True

    Init_Epoch = config.Init_epoch
    Freeze_Epoch = config.Freeze_epoch
    Freeze_batch_size = config.batch_size

    UnFreeze_Epoch = config.epoch
    UnFreeze_batch_size = 16

    Freeze_train = True

    learning_rate = 1e-2
    min_learning_rate = learning_rate*0.01

    optimizerType = 'sgd'
    momentum = 0.937
    weight_decay = 5e-4

    # 学习率下降方式 'step'、'cos'
    lr_decay_type = 'cos'

    log_dir = config.logdir
    save_weight_name = config.save_weight

    train_txt = config.train_txt
    val_txt = config.val_txt

    class_names = get_classes(classes_path=classes_path)
    num_classes = len(class_names)

    # 创建模型
    model = yolo_body([None, None, 3], num_classes = num_classes, phi = phi, weight_decay=weight_decay)
    # 加载预训练权重
    model.load_weights(pretrain_model_path, by_name=True, skip_mismatch=True)
    print('success load pretrain model.')

    model = get_yolox_model(model, input_shape, num_classes)
    

    if Freeze_train:
        freeze_layers = {'tiny': 125, 's': 125, 'm': 179, 'l': 234, 'x': 290}[phi]
        for i in range(freeze_layers):
            model.layers[i].trainable = False
        print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))
        batch_size = Freeze_batch_size if Freeze_train else UnFreeze_batch_size
        # 设置学习率的参数
        nbs     = 64
        Init_lr_fit = max(batch_size / nbs * learning_rate, 3e-4)
        Min_lr_fit  = max(batch_size / nbs * min_learning_rate, 3e-6)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        # 加载数据
        with open(train_txt, encoding='utf-8') as f:
            train_line = f.readlines()
        with open(val_txt, encoding='utf-8') as f:
            val_line = f.readlines()
        num_train = len(train_line)
        num_val = len(val_line)
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        train_dataloader    = YoloDatasets(train_line, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, mosaic = mosaic, train = True)
        val_dataloader      = YoloDatasets(val_line, input_shape, batch_size, num_classes, Init_Epoch, UnFreeze_Epoch, mosaic = False, train = False)

        optimizer = {
            'adam':Adam(learning_rate=learning_rate, beta_1=momentum),
            'sgd':SGD(learning_rate=learning_rate,momentum=momentum, nesterov=True)
        }[optimizerType]
        
        # 训练参数设置
        model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

        # callback设置
        weight_name = log_dir+save_weight_name
        logging = TensorBoard(log_dir)
        early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
        checkpoint = ModelCheckpoint(weight_name, monitor = 'val_loss', save_weights_only = True, save_best_only = False)
        lr_schedule = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks = [logging, lr_schedule, checkpoint, early_stopping]

        # 训练模型
        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
                    generator           = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = Freeze_Epoch,
                    initial_epoch       = Init_Epoch,
                    callbacks           = callbacks)
        
        # 解冻
        for i in range(len(model.layers)): 
            model.layers[i].trainable = True
        
        batch_size = UnFreeze_batch_size
        nbs     = 64
        Init_lr_fit = max(batch_size / nbs * learning_rate, 3e-4)
        Min_lr_fit  = max(batch_size / nbs * min_learning_rate, 3e-6)
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
        callbacks       = [logging, checkpoint, lr_scheduler, early_stopping]

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        train_dataloader.batch_size    = batch_size
        val_dataloader.batch_size      = batch_size

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
        model.fit_generator(
                    generator           = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = Freeze_Epoch,
                    initial_epoch       = UnFreeze_Epoch,
                    callbacks           = callbacks)
        
        # 以Tensorflow格式保存模型
        model.save('./model/village_yolox', save_format='tf2')



        
        
        




        




