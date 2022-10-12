# YoloSeries

实现YOLO系列的算法，当前实现的算法有：

- [x] YOLOV4
- [x] YOLOV5
- [x] YOLOX
- [x] YOLOV7
- [ ] YOLOP

后续将追踪关于YOLO的一些成果，并及时复现当中的算法。具体可以参考本人的博客：[目标检测2022最新进展](https://blog.csdn.net/u012655441/article/details/123552537)。关于YOLO的部署可以参考我的仓库：[Deployment](https://github.com/RyanCCC/Deployment)。

## 仓库说明

文件结构：

``` python

+---Attention： 实现注意力机制
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
|---data：anchor的文件、字体等。
|---lib：基础模块，如数据加载模块，损失函数设置模块。
|---net：网络算法的实现
\---predict.py：推理实现脚本
```


## 执行步骤：

1. 生成训练集、测试集以及验证集：运行voc_annotation.py，注意路径设置

```python
base_path = './villages'
class_file = './villages/village.names'

数据集的文件结构：

|---train_datasets
    +---Annotations
    +---ImageSets
        |---Main
           |---test.txt
           |---train.txt
           |---trainval.txt
           |---val.txt
    \---JPEGImages

```

2. 运行`train.py`文件，要注意一些路径的设置，另外`train.py`里有个开关，选择`YOLOX`还是`YOLOV4`的算法，当然你也改造成在命令行中用`argument`做选择。下面YOLOX同理。

```python
annotation_path = './train_datasets/train.txt'
log_dir = 'logs/'
classes_path = 'train_datasets/village.names'    
anchors_path = './data/yolo_anchors.txt'
weights_path = './data/yolo4_weight.h5'
save_model_name = 'village.h5'
input_shape = (416,416)
```

## MAP计算步骤

1. 统计测试集的groundtrue

```python

python get_gt_txt.py

```

2. 计算模型推理测试集的结果

```python

python get_dr_txt.py

```

3. 计算map的性能指标

```python

python get_map.py

```

想要查看更多资料，可参考[Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)


## YOLOV4

**论文：**[YOLOV4论文](https://arxiv.org/pdf/2004.10934.pdf)

**代码：**[YOLOV4代码实现](https://github.com/AlexeyAB/darknet)

**网络结构：**

![image](https://user-images.githubusercontent.com/27406337/178643886-4602cfc9-ccc3-4a87-8b59-76e80e18cc65.png)

更多的YOLOV4资料可以参考：[yolov4](./doc/yolov4.md)、[目标检测算法Yolov4详解](https://cloud.tencent.com/developer/article/1748630) 

## YOLOX

**论文：**[YOLOX Paper](https://arxiv.org/abs/2107.08430)

**代码：** [YOLOX Code](https://github.com/Megvii-BaseDetection/YOLOX)

**网络结构：**

![image](https://user-images.githubusercontent.com/27406337/178643992-8d3149e3-b54b-4949-81db-e7baf94cbf27.png)

YOLOX的代码可以直接运行```train_yolox.py```即可。

## YOLOV5

**代码：**[YOLOV5 Code](https://github.com/ultralytics/yolov5)

## YOLOV7

**论文：**[YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/abs/2207.02696)

**代码：**[CODE:yolov7](https://github.com/WongKinYiu/yolov7)

**模型结构**

![image](https://user-images.githubusercontent.com/27406337/192188598-0d3fd9e4-7c76-48b2-af0f-2aa3fc972661.png)

**性能**

![image](https://user-images.githubusercontent.com/27406337/192188766-30bf2071-d6a4-4e92-a617-842f7b7f05ab.png)

![image](https://user-images.githubusercontent.com/27406337/192188827-9bf1f99c-8754-4032-8375-9f39277ed972.png)

![image](https://user-images.githubusercontent.com/27406337/192188811-78912328-a279-4127-9be6-9bbb071a035f.png)

## 模型转换

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

注意：使用权重模型的时候要在`customerConfig.py`文件中核实类别文件和anchor文件是否配置正确。另外后续需要导出成TensorRT的Engine模型或者Openvino的模型可以自行定义。当前的参数已经足以使用，后续假设㓟更多参数需求会持续更新优化。

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

## 参考

1. [Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)

2. [mAP计算代码](https://github.com/Cartucho/mAP)

3. [YOLOV3-tf2](https://github.com/zzh8829/yolov3-tf2)


