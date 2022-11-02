from torchvision.models import resnet18,resnet50
from thop import profile
import torch
model = resnet50()
input = torch.randn(1, 3, 224, 224) #模型输入的形状,batch_size=1
flops, params = profile(model, inputs=(input, ))
print(flops/1e9,params/1e6) #flops单位G，para单位M