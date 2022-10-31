# YoloSeries

本仓库主要实现`YOLO`系列的算法，主要是基于`Tensorflow2`或者`Pytorch`实现。特别说明一下，仓库代码中百分之一百会存在bug，欢迎参与到本仓库的建设或者提issue。当前实现的算法有：

- [x] YOLOV4
- [x] YOLOV5
- [x] YOLOX
- [x] YOLOV7
- [ ] YOLOP

本仓库主要是算法的训练和算法的验证，后续的工程化应用部署可以参考我的另一个仓库：[Deployment](https://github.com/RyanCCC/Deployment)。部署仓库包括`python`或者`C++`的部署代码以及`OpenCV`的推理、`TensorRT`的推理，喜欢的话可以给个star或者一起参与建设仓库。毕竟人多力量大。

## 项目结构

``` python
+---Attention： 实现注意力机制
|---cfg: 存放配置文件。
|---tools:存放工具：包括生成训练文档、类别统计等。
|---deep_sort：目标跟踪Deepsort算法
|---doc：存放YOLO资料文档
|---evaluate：存放模型评估方法
|---font：字体
|---logs：存放训练日志的文档
|---model：存放模型和权重
|   +---yolo4_voc_weights.h5：VOC预训练权重
|   \---yolo4_weight.h5：COCO预训练权重
|---result：推理结果保存的文件夹
|---train：训练文档，
|   +---train_tiny.py：YOLOV4 TINY训练
|   +---train_yolox.py：YOLOX训练
|   \---train.py：YOLOV4训练
|---utils：基础模块，如配置文件，训callbacks练设置以及一些jupty notebook
|---video：视频存放
|---yolop：yolop实现算法
|---yolov4：yolov4实现算法
|---yolov5：yolov5实现算法
|---yolov7：yolov7实现算法
|---yolox：yolox实现算法
\---datasets：数据集，以VOC数据集格式
    +---Annotations：数据集标注
    +---ImageSets
    |   \---Main：划分训练集、测试集、验证集的txt文档
    \---JPEGImages：图像
```

每个算法文件夹下的结构如下，以`yolov4`为例：

```
+---checkppoints： 存放训练过程中的checkpoints
|---data：anchor的文件等。
|---lib：基础模块，如数据加载模块，损失函数设置模块。
|---net：网络算法的实现
\---predict.py：推理实现脚本
```


## 数据集制作

数据集的格式按照VOC格式的数据集。主要的文件架构如下所示：

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
D:\villages\JPEGImages/wc_102.jpg 261,607,341,778,0
D:\villages\JPEGImages/camera_20210817154550.jpg 94,149,165,251,11 22,164,97,249,11
D:\villages\JPEGImages/pavilion_20210812155531.jpg 111,2,458,337,10
```

**注意两者之前生成顺序**：先划分数据集，生成`ImageSets/Main`下的txt文档，然后根据`ImageSets/Main`下的txt文档生成适用于文档格式的数据集。

## 模型训练

在模型开始训练之前，先在`cfg`文档下找到对应的算法配置文件进行修改，修改内容包括数据集路径、一些学习参数等。修改完之后执行：

`python train.py --model YOLOX`。即可执行`YOLOX`算法的训练。其中`--model`参数默认为`YOLOV5`，可选的算法有`YOLOV4`，`YOLOV4-TINY`、`YOLOV5`、`YOLOX`，`YOLOV7`等。后续会继续优化，根据需要添加更多的参数。

## 模型推理

模型推理查看`predict.py`。参数说明：
- model：选择推理算法
- show：是否展示图像，不建议
- save：是否保存推理结果
- save_dir：保存推理结果的文件夹位置
- img_dir：推理图像所在的文件夹，批量推理的形式
- weights：算法权重
- image：推理单张图像
- model：批量文件夹推理还是单张图像推理，可取值有：`dir`和`image`

推理的示范：

```sh
python .\predict.py --model YOLOV5 --mode dir --img_dir ./samples --weights ./model/village2022_yolov5_l_20221013.h5 --save
```

```sh
python .\predict.py --model YOLOV7-TINY --img_dir .\samples\ --weights .\model\village_Detection_yolov7_tiny_2022_10_27.pth --mode dir --save
```

## 模型验证

1. 统计测试集的groundtrue

```python
python ./evaluate/get_gt_txt.py --testset ./VOC2017/ImageSets/Main/test.txt --annotation ./villages/Annotations/ --gt_folder ./result/gt_folder
```

参数说明：
- testset：划分测试数据集保存的`txt`文档
- annotation：标注文件保存的文件夹
- gt_folder：保存文件夹路径


2. 计算模型推理测试集的结果

```python
python ./evaluate/get_dr_txt.py --testset ./villages/ImageSets/Main/test.txt --pr_folder ./result/pr_folder --minoverlap 0.5 --model_path ./model/village_yolox.h5 --image_path ./villages/JPEGImages/ --model YOLOX
```

参数说明：
- testset：划分测试数据集保存的`txt`文档
- pr_folder：推理结果保存的文件夹，保存文件格式：`txt`。
- model_path：推理模型的权重
- image_path：图像路径
- model：使用的算法模型

3. 计算map的性能指标

```python
python get_map.py
```


资料可参考[Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)

## 模型部署

### 模型优化

模型的优化包括模型的量化和剪枝。模型的量化与剪枝都是通过tensorflow简单的API进行操作，还没有进行深入的研究。

### 模型转换

模型转换主要应用在生产环境中，关于模型转换用了YOLOX作为例子，详情可以参考：[TF2ONNX](https://github.com/RyanCCC/Deployment/tree/main/ONNXDemo/Tensorflow)，当中有YOLOX转换成ONNX的例子。转换成ONNX之后需后续就可以为所欲为了，比如需要部署到TensorRT或者Openvino中等都可以通过ONNX转换成对应的格式的模型。模型转换之后至于模型的性能，如精确度、速度等有没有损失，在此我没有做相应的测试，感兴趣的可以自行测评一下模型性能差异。

模型导出脚本：`export.py`，相关参数说明如下：

- `weight`：YOLO模型的权重路径，用于从权重路径中导出pb模型或者ONNX模型
- `saved_pb`：是否保存Tensorfow的PB模型，带上这个参数后需要指定`saved_pb_dir`参数，表示模型保存的路径。
- `saved_pb_dir`：保存PB模型的路径。
- `yolo`：选择需要导出的YOLO算法。
- `saved_model`：该参数用于直接加载Tensorflow的PB模型，并导出成ONNX模型。
- `save_onnx`：ONNX模型保存的路径。
- `opset`：ONNX的算子类型，默认12
- `flag`：带上这个参数表示从Tensorflow的PB模型进行导出，否则从权重中导出。

注意：使用权重模型的时候要在`cfg`目录下对应的配置文件中核实类别文件和anchor文件是否配置正确。另外后续需要导出成TensorRT的Engine模型或者Openvino的模型可以自行定义。当前的参数已经足以使用，后续假设㓟更多参数需求会持续更新优化。

使用例子：

1. 从Tensorflow模型导出：

```
python .\export.py --saved_model .\village_model\ --save_onnx './tmp.onnx' --yolo yolox --flag
```

2. 从Tensorflow权重导出：

记住：**一定要先在配置文件中配置好模型再进行导出！**
```
python .\export.py --yolo yolox --weight .\model\village_yolox.h5 --save_onnx './tmp_yolox.onnx'
```

更多的ONNX推理和算法部署可参考：[Deployment](https://github.com/RyanCCC/Deployment)。

## 相关资料

### YOLOV4

**论文：**[YOLOV4论文](https://arxiv.org/pdf/2004.10934.pdf)

**代码：**[YOLOV4代码实现](https://github.com/AlexeyAB/darknet)

**网络结构：**

![image](https://user-images.githubusercontent.com/27406337/178643886-4602cfc9-ccc3-4a87-8b59-76e80e18cc65.png)

更多的YOLOV4资料可以参考：[yolov4](./doc/yolov4.md)、[目标检测算法Yolov4详解](https://cloud.tencent.com/developer/article/1748630) 

### YOLOX

**论文：**[YOLOX Paper](https://arxiv.org/abs/2107.08430)

**代码：** [YOLOX Code](https://github.com/Megvii-BaseDetection/YOLOX)

**网络结构：**

![image](https://user-images.githubusercontent.com/27406337/178643992-8d3149e3-b54b-4949-81db-e7baf94cbf27.png)

YOLOX的代码可以直接运行```train_yolox.py```即可。

### YOLOV5

**代码：**[YOLOV5 Code](https://github.com/ultralytics/yolov5)

### YOLOV7

**论文：**[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

**代码：**[CODE:yolov7](https://github.com/WongKinYiu/yolov7)

**模型结构**

![image](https://user-images.githubusercontent.com/27406337/192188598-0d3fd9e4-7c76-48b2-af0f-2aa3fc972661.png)

**性能**

![image](https://user-images.githubusercontent.com/27406337/192188766-30bf2071-d6a4-4e92-a617-842f7b7f05ab.png)

![image](https://user-images.githubusercontent.com/27406337/192188827-9bf1f99c-8754-4032-8375-9f39277ed972.png)

![image](https://user-images.githubusercontent.com/27406337/192188811-78912328-a279-4127-9be6-9bbb071a035f.png)


## 参考

1. [Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)

2. [mAP计算代码](https://github.com/Cartucho/mAP)

3. [YOLOV3-tf2](https://github.com/zzh8829/yolov3-tf2)

