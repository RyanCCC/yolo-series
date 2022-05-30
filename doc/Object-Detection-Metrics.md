## 目标检测评价指标


可参考以下仓库：https://github.com/rafaelpadilla/Object-Detection-Metrics

### 准确率（Accuracy）

  正确分类的样本数除以样本总数，即 accuracy=正确预测的正反例数

### 错误率（Error rate）

  错误率与正确率相反，描述被分类器错分的比例，即：
    
  错误率 = 1-准确率

### 混淆矩阵（Confusion Matrix）

  混淆矩阵又被称为错误矩阵，在每个类别下，模型预测错误的结果数量以及错误预测类别和正确预测的数量都在这一矩阵下面显示出来。
    
  ![image](https://user-images.githubusercontent.com/27406337/130168328-e89ee068-6726-4626-a952-c07b051c374a.png)

### 召回率（Recall）
    
    查全率，预测为正例的样本中正确的数量除以真正的Positive的数量，即Recall = TP/(TP+FN) = TP/P

### 精确率（Precision）

  查准率，被分为正例的示例中实际为正例的比例，即：Precision = TP/(TP+FP)
    
  ![image](https://user-images.githubusercontent.com/27406337/130168701-b8da8275-5046-413e-b462-1fc63310be08.png)

### PR曲线

  选取不同阈值时对应的精度和召回。总体趋势，精度越高，召回越低，当召回达到1时，对应概率分数最低的正样本。
    
  ![image](https://user-images.githubusercontent.com/27406337/130168796-682d5e7a-f525-4024-bdba-de80f7721f58.png)

### 平均精度（Average-Precision AP）

  P-R曲线围起来的面积

### F指标（F measure）

  ![image](https://user-images.githubusercontent.com/27406337/130169188-bbe0aa90-d67a-4fcf-ae41-e2aa45eb7955.png)


### ROC曲线
    
  True Positive Rate ( TPR ) = TP / [ TP + FN] ，TPR代表能将正例分对的概率
    
  False Positive Rate( FPR ) = FP / [ FP + TN] ，FPR代表将负例错分为正例的概率
   
  ![image](https://user-images.githubusercontent.com/27406337/130169395-b392bdb6-9b46-4409-a266-f1c6a014acca.png)
    
  AUC则是ROC曲线围住的面积。