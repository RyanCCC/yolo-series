# YOLO目标检测算法

本仓库主要实现`YOLO`目标检测算法，欢迎参与到本仓库的建设或者提issue。本仓库有两个分支，一个分支为`main`，主要是基于`Tensorflow`实现yolo算法，另一个分支是`pytorch`，主要是基于`pytorch`实现yolo算法。`main`分支当前算法实现基本情况如下：

|    算法     |    实现框架    |               备注                |
| :---------: | :------------: | :-------------------------------: |
|   YOLOV4    | Tensorflow-2.4 | 已实现，并通过测试 |
|   YOLOV5    | Tensorflow-2.4 | 已实现，并经过测试 |
| YOLOV5-v6.1 | Tensorflow-2.4 | 已实现，并通过测试 |
|    YOLOX    | Tensorflow-2.4 | 已实现，并通过测试 |
|   YOLOV7    | Tensorflow-2.4 | 已实现，并通过测试 |

本仓库主要是算法的训练和算法的验证，后续的工程化应用部署可以参考另一个仓库：[Deployment](https://github.com/RyanCCC/Deployment)。部署仓库包括模型的量化与压缩、`python`或者`C++`的部署代码以及`OpenCV`和`TensorRT`等推理，喜欢或者对您有用的话可以给个:star:或者一起参与建设仓库:hand:。

## 项目结构

``` python
+---cfg: 存放配置文件。
|---components： CNN组件，如batch_renorm、注意力机制
|---tools:存放工具：包括生成训练文档、类别统计等。
|---tracking：目标跟踪Deepsort算法
|---doc：存放YOLO资料文档，包括backbone、后处理等算法文档
|---evaluate：存放模型评估方法
|---font：字体
|---logs：存放训练日志的文档
|---model：存放模型和权重
|   +---yolo4_voc_weights.h5：VOC预训练权重
|   \---yolo4_weight.h5：COCO预训练权重
|---result：推理结果保存的文件夹
|---video：视频保存文件夹
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

可按照`VOC`格式的数据集进行制作，主要的文件结构如下所示：

```
+---Annotation： 数据标注,xml格式
|---ImageSets：划分数据集
|   +---Main：存放txt位置
|---JPEGSImages:存放图像
|---labels.names：类别标签文档
```

当然也可以根据实际情况自定义自己的数据集结构，训练的时候只需要修改路径即可。另外在`tools`文件夹下有一些关于数据集脚本：

1. `splitDataset.py`：划分数据集，即产生`ImageSets/Main`下的txt文档
2. `voc_annotation.py`：生成输入到网络的文档格式，如下所示：

```
your/dataset/root/path/JPEGImages/wc_102.jpg 261,607,341,778,0
your/dataset/root/path/camera_20210817154550.jpg 94,149,165,251,11 22,164,97,249,11
your/dataset/root/path/pavilion_20210812155531.jpg 111,2,458,337,10
```

**注意两者之前生成顺序**：先划分数据集，生成`ImageSets/Main`下的txt文档，然后根据`ImageSets/Main`下的txt文档生成适用于文档格式的数据集。

## 模型训练

在模型开始训练之前，先在`cfg`文档下找到对应的算法配置文件进行修改，修改内容包括数据集路径、训练参数参数等。在配置文件修改完相关配置后，可在每个算法文件下的含有`train`命名的文件进一步检查其他训练参数，如`yolov5`下的`train_yolov5.py`文件里设置的训练参数。

当检查所有参数都没有错误后，可执行训练：

```sh
python train.py --model YOLOV5
```

参数说明：
- **model**：表示训练的算法，如`YOLOV5`、`YOLOX`等。


## 模型推理

模型推理查看`predict.py`。参数说明：
- **yolo**：选择推理算法，如`YOLOV5`、`YOLOV5V61`等算法。
- **show**：在进行文件夹推理的时候是否展示图像，此处不建议。
- **save**：是否保存推理结果。
- **save_dir**：保存推理结果的文件夹位置
- **model**：具体算法模型，可以是算法的权重、ONNX模型。
- **source**：待检测的对象，可以是单张图像，也可以是文件夹

推理的示范：

```sh
# directory inference
python predict.py --yolo yolox --model /your/model/path/voc.h5 --source ./samples/
```

```sh
# image inference
python ./predict.py --yolo YOLOV7-TINY --source ./samples/test.jpg --model ./model/VOC2007_yolov7_tiny_2022_10_28.h5 --save
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
python ./evaluate/get_dr_txt.py --testset ./VOC2007/ImageSets/Main/test.txt --pr_folder ./result/pr_folder --minoverlap 0.5 --model_path ./model/voc_yolox.h5 --image_path ./VOC2007/JPEGImages/ --model YOLOX
```

参数说明：
- **testset**：划分测试数据集保存的`txt`文档
- **pr_folder**：推理结果保存的文件夹，保存文件格式：`txt`。
- **model_path**：推理模型的权重
- **image_path**：图像路径
- **model**：使用的算法模型

3. **计算map的性能指标**

```sh
python ./evaluate/get_map.py --GT_PATH ./result/evaluate --DR_PATH ./result/pr_folder/ --IMG_PATH ./VOC2007/JPEGImages/ --MINOVERLAP 0.5
```

参数说明：

- **GT_PATH**：脚本`get_gt_txt.py`生成文件的保存路径
- **DR_PATH**：脚本`get_dr_txt.py`生成文件的保存路径
- **IMG_PATH**：测试集图像的路径
- **MINOVERLAP**：map@**的值

### FPS计算

在`predict.py`使用`dir`参数进行推理即可获取模型推理的fps。

### FLOPs计算

**FLOPs**：注意`s`小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。**FLOPS**：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。

### PARAMS计算

详情可参考[Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)


## 模型转换

在生产环境中需要进行模型转换，此处主要是将模型转换成ONNX格式模型。后续如果需要再转换成其他格式的模型，如`TensorRT`、`TFLite`等都可通过ONNX转换成对应的格式的模型。模型转换成ONNX的性能，如精确度、速度等是否与原模型存在差异，在此没有做相应的测试:dizzy_face:，感兴趣的可以自行测评一下模型性能差异。

模型导出脚本：`export.py`，相关参数说明如下：

- **model**：模型的权重路径
- **saved_pb**：是否保存`pb`格式模型，带上这个参数后需要指定`saved_pb_dir`参数，表示模型保存的路径。
- **saved_pb_dir**：保存`pb`模型的路径。
- **yolo**：选择需要导出的YOLO算法。
- **save_onnx**：ONNX模型保存的路径。
- **opset**：ONNX的算子类型，默认12

注意：使用权重模型的时候要在`cfg`目录下对应的配置文件中核实类别文件和anchor文件是否配置正确。另外后续需要导出成TensorRT的Engine模型或者Openvino的模型可以自行定义。当前的参数已经足以使用，后续假设㓟更多参数需求会持续更新优化。

使用例子：

``` sh
python ./export.py --model ./model/VOC.h5 --yolo yolox --save_onnx 'voc_yolox_l_13_640_v1.onnx' 
```

更多的ONNX推理和算法部署可参考：[Deployment](https://github.com/RyanCCC/Deployment)。


## 参考

更多算法的内容可以参考`doc`文件下的文档。

1. [Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)
2. [mAP计算代码](https://github.com/Cartucho/mAP)
3. [YOLOV3-tf2](https://github.com/zzh8829/yolov3-tf2)

