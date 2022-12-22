# YOLO目标检测算法

本仓库基于`Pytorch`实现`YOLO`目标检测算法，欢迎参与到本仓库的建设或者提issue。当前算法实现基本情况如下：

|    算法     |    实现框架    |               备注                |
| :---------: | :------------: | :-------------------------------: |
|   YOLOV4    |   Torch-1.9    | 已实现，并通过测试。  |
|   YOLOV5    |   Torch-1.9    | 已实现，并通过测试。 |
| YOLOV5-v6.1 |   Torch-1.9    | 已实现，并通过测试。 |
|    YOLOX    |   Torch-1.9    | 已实现，并通过测试。  |
|   YOLOV7    |   Torch-1.9    | 已实现，并通过测试。|
|    YOLOP    |   Torch-1.9    | 已实现，未通过测试。 |

关于YOLO算法的工程化应用部署可以参考另一个仓库：[Deployment](https://github.com/RyanCCC/Deployment)。部署仓库包括模型的量化与压缩、`python`或者`C++`的部署代码以及`OpenCV`和`TensorRT`等推理，喜欢的话可以给个:star:或者一起参与建设仓库:hand:。

## 项目结构

``` python
+---cfg: 存放配置文件。
|---tools:存放工具：包括生成训练文档、类别统计等。
|---tracking：目标跟踪Deepsort算法
|---evaluate：模型评估方法
|---font：字体
|---logs：存放训练日志的文档
|---model：存放模型和权重
|   +---yolo4_voc_weights.pth：VOC预训练权重
|   \---yolo4_weight.pth：COCO预训练权重
|---result：推理结果保存的文件夹
|---video：视频保存文件夹
|---yolop：yolop算法实现
|---yolov4：yolov4算法实现
|---yolov5：yolov5算法实现
|---yolov7：yolov7算法实现
|---yolox：yolox算法实现
\---datasets：数据集，以VOC数据集格式
    +---Annotations：数据集标注
    +---ImageSets
    |   \---Main：划分训练集、测试集、验证集的txt文档
    \---JPEGImages：图像
```


## 数据集制作

可按照`VOC`格式制作你的数据集，主要的文件结构如下所示：

```
+---Annotation： 数据标注,xml格式
|---ImageSets：划分数据集
|   +---Main：存放txt位置
|---JPEGSImages:存放图像
|---labels.names：类别标签文档
```

当然你也可以自定义自己的数据集结构，训练的时候只需要修改路径即可。另外在`tools`文件夹下有一些关于数据集脚本：

1. `splitDataset.py`：划分数据集，即产生`ImageSets/Main`下的txt文档
2. `voc_annotation.py`：生成输入到网络的文档格式，如下所示：

```
your/voc/root/path/JPEGImages/wc_102.jpg 261,607,341,778,0
your/voc/root/path/JPEGImages/camera_20210817154550.jpg 94,149,165,251,11 22,164,97,249,11
your/voc/root/path/JPEGImages/pavilion_20210812155531.jpg 111,2,458,337,10
```

**注意两者之前生成顺序**：先划分数据集，生成`ImageSets/Main`下的txt文档，然后根据`ImageSets/Main`下的txt文档生成适用于文档格式的数据集。

## 模型训练

在开始训练模型之前在`cfg`文档下找到对应的算法配置文件进行修改，修改内容包括数据集路径、训练参数参数等。修改完相关配置，可在每个算法文件下的含有`train`命名的文件进一步检查训练参数，如`yolov5`下的`train_yolov5.py`文件设置的训练参数。当确定好训练参数，数据集路径后，执行以下命令即可进行训练。

```sh
python train.py --model YOLOV5
```

参数说明：

- **model**：表示训练的算法，如`YOLOV5`、`YOLOX`等。

### 训练说明

1. 训练前检查自己的图像格式。图像为`.jpg`格式，标签为`.xml`格式。图像在训练前的预处理会自动进行`resize`，灰度图会自动转换成RGB图像。
2. 训练文档（每个算法下的含有`train`的文档）中的`distributed`参数是用于指定是否使用单机多卡分布式运行。设置该参数为`True`后，可以使用`sync_bn`。
3. `fp16`参数是表示是否使用混合精度训练，可减少约一半的内存，但需要pytorch1.7.1以上。
4. 关于数据增强的参数。`mosaic`表示是否需要马赛克增强，`mosaic_prob`每个step有多少概率使用mosaic数据增强，默认为0.5。`mixup`表示是否使用mixup数据增强，当使用mosaic时才有效，表示会对mosaic增强后的图像进行mixup处理。`mixup_prob`表示多少概率使用mixup增强，默认为0.5。`special_aug_ratio`：由于Mosaic生成的训练图片，远远脱离自然图片的真实分布。当`mosaic=True`时，`special_aug_ratio`范围内进行`mosaic`数据增强。
5. 训练分为两个阶段，分别是冻结阶段和解冻阶段。设置冻结阶段是为了满足机器性能不足的训练需求。
6. 关于优化器的参数`optimizer_type`可选的有adam、sgd。当使用Adam优化器时建议设置 `Init_lr=1e-3`，当使用SGD优化器时建议设置`Init_lr=1e-2`。`momentum`表示优化器内部使用到的momentum参数。`weight_decay`表示权值衰减，可防止过拟合。`adam`会导致`weight_decay`错误，使用`adam`优化器时建议设置为0。
7. 关于验证参数`eval_flag`表示是否在训练时进行评估，评估对象为验证集。

## 模型推理

模型推理查看`predict.py`。参数说明：
- **model**：选择推理算法
- **save**：是否保存推理结果
- **save_dir**：保存推理结果的文件夹位置
- **weights**：算法权重
- **source**：推理单张图像或者文件夹

运行样例：

```sh
# dir inference
python ./predict.py --model YOLOV7 --source ./samples/ --weights ./model/voc_2007.pt --save --save_dir ./result/ 
```

```sh
# image inference
python ./predict.py --model YOLOV5-V61 --source ./samples/images/1.jpg --weights ./model/voc_2007.onnx --save --save_dir ./result/
```

## 模型验证

### MAP计算

1. **统计测试集的ground_True**

```sh
python ./evaluate/get_gt_txt.py --testset ./VOC2017/ImageSets/Main/test.txt --annotation ./VOC2007/Annotations/ --gt_folder ./result/gt_folder
```

参数说明：
- **testset**：划分测试数据集保存的`txt`文档
- **annotation**：标注文件保存的文件夹
- **gt_folder**：保存文件夹路径


2. **计算模型推理测试集的结果**

```sh
python ./evaluate/get_dr_txt.py --testset ./voc2007/ImageSets/Main/test.txt --pr_folder ./result/pr_folder --minoverlap 0.5 --model_path ./model/voc2007_yolox.h5 --image_path ./voc2007/JPEGImages/ --model YOLOX
```

参数说明：
- **testset**：划分测试数据集保存的`txt`文档
- **pr_folder**：推理结果保存的文件夹，保存文件格式：`txt`。
- **model_path**：推理模型的权重
- **image_path**：图像路径
- **model**：使用的算法模型

3. **计算map的性能指标**

```sh
python ./evaluate/get_map.py --GT_PATH ./result/evaluate --DR_PATH ./result/pr_folder/ --IMG_PATH ./voc2007/JPEGImages/ --MINOVERLAP 0.5
```

参数说明：

- **GT_PATH**：脚本`get_gt_txt.py`生成文件的保存路径
- **DR_PATH**：脚本`get_dr_txt.py`生成文件的保存路径
- **IMG_PATH**：测试集图像的路径
- **MINOVERLAP**：map@**的值

### FPS计算

在`predict.py`使用`dir`参数进行推理即可获取模型推理的fps。

### FLOPs计算

**FLOPs**：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。**FLOPS**：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。

### PARAMS计算

详情可参考[Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)


### 模型转换

在生产环境中需要进行模型转换，此处主要是将模型转换成ONNX格式模型。后续如果需要再转换成其他格式的模型，如`TensorRT`、`TFLite`等都可通过ONNX转换成对应的格式的模型。模型转换成ONNX的性能，如精确度、速度等是否与原模型存在差异，在此没有做相应的测试😵，感兴趣的可以自行测评一下模型性能差异。

模型导出脚本：`export.py`，相关参数说明如下：

- `weight`：YOLO模型的权重路径，用于从权重路径中导出模型
- `yolo`：选择需要导出的YOLO算法。
- `save_file`：ONNX模型保存的路径。
- `dynamic`：onnx的动态输入
- `opset`：ONNX的算子类型，默认12
- `simplify`：ONNX简化模型，一般为不简化。

**注意**：使用权重模型的时候要在`cfg`目录下对应的配置文件中核实类别文件和anchor文件是否配置正确。另外后续需要导出成TensorRT的Engine模型或者Openvino的模型可以自行定义。当前的参数已经足以使用，后续假设㓟更多参数需求会持续更新优化。

使用例子：

```sh
python ./export.py --weight ./voc_yolov5_l.pth --save_file ./yolov5_l_12.onnx --yolo yolov5
```

