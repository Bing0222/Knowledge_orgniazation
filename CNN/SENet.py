# @Time    : 2023/2/19
# @Author  : Bing

# -------------------------------------------- #
# (1) Squeeze-and-Excitation Networks (SE注意力机制)
# General SE put behind down sample
# -------------------------------------------- #

'''
1）Squeeze：通过全局平均池化，将每个通道的二维特征（H*W）压缩为1个实数，将特征图从 [h, w, c] ==> [1,1,c]
2）excitation：给每个特征通道生成一个权重值，论文中通过两个全连接层构建通道间的相关性，输出的权重值数目和输入特征图的通道数相同。[1,1,c] ==> [1,1,c]
3）Scale：将前面得到的归一化权重加权到每个通道的特征上。论文中使用的是乘法，逐通道乘以权重系数。[h,w,c]*[1,1,c] ==> [h,w,c]


1）SENet的核心思想是通过全连接网络根据loss损失来自动学习特征权重，而不是直接根据特征通道的数值分配来判断，使有效的特征通道的权重大。当然SE注意力机制不可避免的增加了一些参数和计算量，但性价比还是挺高的。
2）论文认为excitation操作中使用两个全连接层相比直接使用一个全连接层，它的好处在于，具有更多的非线性，可以更好地拟合通道间的复杂关联。

'''

import torch
from torch import nn
from torchstat import stat  # 查看网络参数


# 定义SE注意力机制的类
class se_block(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接下降通道的倍数
    def __init__(self, in_channel, ratio=4):
        # 继承父类初始化方法
        super(se_block, self).__init__()

        # 属性分配
        # 全局平均池化，输出的特征图的宽高=1
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # 第一个全连接层将特征图的通道数下降4倍
        self.fc1 = nn.Linear(in_features=in_channel, out_features=in_channel // ratio, bias=False)
        # relu激活
        self.relu = nn.ReLU()
        # 第二个全连接层恢复通道数
        self.fc2 = nn.Linear(in_features=in_channel // ratio, out_features=in_channel, bias=False)
        # sigmoid激活函数，将权值归一化到0-1
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):  # inputs 代表输入特征图

        # 获取输入特征图的shape
        b, c, h, w = inputs.shape
        # 全局平均池化 [b,c,h,w]==>[b,c,1,1]
        x = self.avg_pool(inputs)
        # 维度调整 [b,c,1,1]==>[b,c]
        x = x.view([b, c])

        # 第一个全连接下降通道 [b,c]==>[b,c//4]
        x = self.fc1(x)
        x = self.relu(x)
        # 第二个全连接上升通道 [b,c//4]==>[b,c]
        x = self.fc2(x)
        # 对通道权重归一化处理
        x = self.sigmoid(x)

        # 调整维度 [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])

        # 将输入特征图和通道权重相乘
        outputs = x * inputs
        return outputs


# 构造输入层shape==[4,32,16,16]
inputs = torch.rand(4, 32, 16, 16)
# 获取输入通道数
in_channel = inputs.shape[1]
# 模型实例化
model = se_block(in_channel=in_channel)

# 前向传播查看输出结果
outputs = model(inputs)
print(outputs.shape)  # [4,32,16,16])

print(model)  # 查看模型结构
stat(model, input_size=[32, 16, 16])  # 查看参数，不需要指定batch维度