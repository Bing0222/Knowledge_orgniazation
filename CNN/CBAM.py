# @Time    : 2023/2/20
# @Author  : Bing

import torch
from torch import nn
from torchstat import stat

"""
CBAM注意力机制是由通道注意力机制（channel）和空间注意力机制（spatial）组成
"""

'''
channel(通道注意力机制)
先将输入特征图分别进行全局最大池化和全局平均池化，对特征映射基于两个维度压缩，获得两张不同维度的特征描述。
池化后的特征图共用一个多层感知器网络，先通过一个全连接层下降通道数，再通过另一个全连接恢复通道数。
将两张特征图在通道维度堆叠，经过 sigmoid 激活函数将特征图的每个通道的权重归一化到0-1之间。将归一化后的权重和输入特征图相乘。
'''
class channel_attention(nn.Module):
    def __init__(self,in_channel,ratio=4):
        super(channel_attention, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        self.avg_pool = nn.AdaptiveMaxPool2d(output_size=1)

        self.fc1 = nn.Linear(in_features=in_channel,out_features=in_channel//ratio,bias=False)
        self.fc2 = nn.Linear(in_features=in_channel//ratio,out_features=in_channel,bias=False)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        b,c,h,w = inputs.shape

        max_pool = self.max_pool(inputs)
        avg_pool = self.avg_pool(inputs)

        max_pool = max_pool.view([b,c])
        avg_pool = avg_pool.view([b,c])

        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        x = x_maxpool + x_avgpool
        x = self.sigmoid(x)
        x = x.view([b,c,1,1])
        outputs = inputs * x

        return outputs


'''
spatial(空间注意力通道)
首先，对输入特征图在通道维度下做最大池化和平均池化，将池化后的两张特征图在通道维度堆叠。
然后，使用 7*7 （或3*3、1*1）大小的卷积核融合通道信息，特征图的shape从 [b,2,h,w] 变成 [b,1,h,w]。
最后，将卷积后的结果经过 sigmoid 函数对特征图的空间权重归一化，再将输入特征图和权重相乘。
'''

class spatial_attention(nn.Module):
    def __init__(self,kernel_size = 7):
        super(spatial_attention, self).__init__()
        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        #  [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool,_ = torch.max(inputs,dim=1,keepdim=True)
        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs,dim=1,keepdim=True)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool,x_avgpool],dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs

'''
入特征图先经过通道注意力机制，将通道权重和输入特征图相乘后再送入空间注意力机制，将归一化后的空间权重和空间注意力机制的输入特征图相乘，得到最终加权后的特征图。
'''
class cbdam(nn.Module):
    def __init__(self,in_channel,ratio=4,kernel_size=7):
        super(cbdam, self).__init__()

        self.channel_attention = channel_attention(in_channel=in_channel,ratio=ratio)
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

    def forward(self,inputs):
        x = self.channel_attention(inputs)
        x = self.spatial_attention(x)
        return x

inputs = torch.rand([4,32,16,16])
in_channel = inputs.shape[1]

model = cbdam(in_channel=in_channel)
outputs = model(inputs)

print(outputs.shape)
print(model)
stat(model,input_size=[32,16,16])