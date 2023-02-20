# @Time    : 2023/2/20
# @Author  : Bing



# --------------------------------------------------------- #
#（2）ECANet 通道注意力机制
# 使用1D卷积代替SE注意力机制中的全连接层
#---------------------------------------------------------  #
'''
1）将输入特征图经过全局平均池化，特征图从 [h,w,c] 的矩阵变成 [1,1,c] 的向量
2）根据特征图的通道数计算得到自适应的一维卷积核大小 kernel_size
3）将 kernel_size 用于一维卷积中，得到对于特征图的每个通道的权重
4）将归一化权重和原输入特征图逐通道相乘，生成加权后的特征图
'''

import torch
from torch import nn
import math
from torchstat import stat

class eca_block(nn.Module):
    def __init__(self,in_channel,b=1,gama=2):
        super(eca_block, self).__init__()
        kernel_size = int(abs((math.log(in_channel,2)+b)/gama))
        if kernel_size % 2:
            kernel_size = kernel_size
        else:
            kernel_size = kernel_size + 1

        padding = kernel_size // 2
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.conv = nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,bias=False,padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整，变成序列形式 [b,c,1,1]==>[b,1,c]
        x = x.view([b, 1, c])
        # 1D卷积 [b,1,c]==>[b,1,c]
        x = self.conv(x)
        # 权值归一化
        x = self.sigmoid(x)
        # 维度调整 [b,1,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘[b,c,h,w]*[b,c,1,1]==>[b,c,h,w]
        outputs = x * inputs
        return outputs


# 构造输入层 [b,c,h,w]==[4,32,16,16]
inputs = torch.rand([4, 32, 16, 16])
# 获取输入图像的通道数
in_channel = inputs.shape[1]
# 模型实例化
model = eca_block(in_channel=in_channel)
# 前向传播
outputs = model(inputs)

print(outputs.shape)  # 查看输出结果
print(model)  # 查看网络结构
stat(model,input_size=[32,16,16])

