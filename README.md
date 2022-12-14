# YOLOSeries

本仓库主要实现`YOLO`目标检测算法，欢迎参与到本仓库的建设或者提issue。当前算法实现基本情况

|    算法     |    实现框架    |               备注                |
| :---------: | :------------: | :-------------------------------: |
|   YOLOV4    | Tensorflow-2.4 | 已实现，并通过测试。`main`分支。  |
|   YOLOV5    |   Torch-1.9    | 已实现，并通过测试。`pytorch`分支。 |
| YOLOV5-v6.1 |   Torch-1.9    | 已实现，并通过测试。`pytorch`分支。 |
|    YOLOX    | Tensorflow-2.4 | 已实现，并通过测试。`main`分支。  |
|   YOLOV7    |   Torch-1.9    | 已实现，并通过测试。`pytorch`分支。|
|    YOLOP    |   Torch-1.9    | 已实现，未通过测试。`main`分支。  |

本仓库主要是算法的训练和算法的验证，后续的工程化应用部署可以参考另一个仓库：[Deployment](https://github.com/RyanCCC/Deployment)。部署仓库包括模型的量化与压缩、`python`或者`C++`的部署代码以及`OpenCV`和`TensorRT`等推理，喜欢的话可以给个star或者一起参与建设仓库。

## 项目结构

``` python
+---Attention： 实现注意力机制
|---cfg: 存放配置文件。
|---tools:存放工具：包括生成训练文档、类别统计等。
|---deep_sort：目标跟踪Deepsort算法
|---doc：存放YOLO资料文档，包括backbone、后处理等算法文档
|---evaluate：存放模型评估方法
|---font：字体
|---logs：存放训练日志的文档
|---model：存放模型和权重
|   +---yolo4_voc_weights.h5：VOC预训练权重
|   \---yolo4_weight.h5：COCO预训练权重
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
your/voc/root/path/JPEGImages/wc_102.jpg 261,607,341,778,0
your/voc/root/pathJPEGImages/camera_20210817154550.jpg 94,149,165,251,11 22,164,97,249,11
your/voc/root/pathJPEGImages/pavilion_20210812155531.jpg 111,2,458,337,10
```

**注意两者之前生成顺序**：先划分数据集，生成`ImageSets/Main`下的txt文档，然后根据`ImageSets/Main`下的txt文档生成适用于文档格式的数据集。

## 模型训练

在模型开始训练之前，先在`cfg`文档下找到对应的算法配置文件进行修改，修改内容包括数据集路径、训练参数参数等。修改完之后执行：`python train.py --model YOLOX`。即可执行`YOLOX`算法的训练。其中`--model`参数默认为`YOLOV5`，可选的算法有`YOLOV4`，`YOLOV4-TINY`、`YOLOV5`、`YOLOX`，`YOLOV7`等。后续会继续优化，根据需要添加更多的算法参数。

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
python ./predict.py --model YOLOV5  --source ./samples --weights ./model/VOC2007_yolov5_l_20221013.h5 --save
```

```sh
python ./predict.py --model YOLOV7-TINY --source ./samples/1.jpg --weights ./model/VOC2007_yolov7_tiny_2022_10_28.pth --save
```

## 模型验证

### MAP计算

1. **统计测试集的ground_True**

```python
python ./evaluate/get_gt_txt.py --testset ./VOC2017/ImageSets/Main/test.txt --annotation ./VOC2007/Annotations/ --gt_folder ./result/gt_folder
```

参数说明：
- testset：划分测试数据集保存的`txt`文档
- annotation：标注文件保存的文件夹
- gt_folder：保存文件夹路径


2. **计算模型推理测试集的结果**

```python
python ./evaluate/get_dr_txt.py --testset ./voc2007/ImageSets/Main/test.txt --pr_folder ./result/pr_folder --minoverlap 0.5 --model_path ./model/voc2007_yolox.h5 --image_path ./voc2007/JPEGImages/ --model YOLOX
```

参数说明：
- testset：划分测试数据集保存的`txt`文档
- pr_folder：推理结果保存的文件夹，保存文件格式：`txt`。
- model_path：推理模型的权重
- image_path：图像路径
- model：使用的算法模型

3. **计算map的性能指标**

```python
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

FLOPs：注意s小写，是floating point operations的缩写（s表复数），意指浮点运算数，理解为计算量。可以用来衡量算法/模型的复杂度。FLOPS：注意全大写，是floating point operations per second的缩写，意指每秒浮点运算次数，理解为计算速度。是一个衡量硬件性能的指标。

### PARAMS计算

详情可参考[Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)


### 模型转换

模型转换主要应用在生产环境中，关于模型转换用了YOLOX作为例子，详情可以参考：[TF2ONNX](https://github.com/RyanCCC/Deployment/tree/main/ONNXDemo/Tensorflow)，当中有YOLOX转换成ONNX的例子。将模型转换成ONNX格式模型后就可以往后继续转换成其他的模型，比如需要部署到TensorRT或者Openvino中等都可以通过ONNX转换成对应的格式的模型。模型转换之后至于模型的性能，如精确度、速度等是否存在差异，在此没有做相应的测试，感兴趣的可以自行测评一下模型性能差异。

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
python .\export.py --saved_model .\voc2007_model\ --save_onnx './tmp.onnx' --yolo yolox --flag
```

2. 从Tensorflow权重导出：

记住：**一定要先在配置文件中配置好模型再进行导出！**
```
python .\export.py --yolo yolox --weight .\model\voc2007_yolox.h5 --save_onnx './tmp_yolox.onnx'
```

更多的ONNX推理和算法部署可参考：[Deployment](https://github.com/RyanCCC/Deployment)。


## 参考

更多算法的内容可以参考`doc`文件下的文档。

1. [Object-Detection-Metrics](./doc/Object-Detection-Metrics.md)
2. [mAP计算代码](https://github.com/Cartucho/mAP)
3. [YOLOV3-tf2](https://github.com/zzh8829/yolov3-tf2)

