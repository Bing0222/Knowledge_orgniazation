"""
构建生成器网络：为每个域设计一个生成器网络，通常使用卷积神经网络（CNN）或类似的结构。
生成器网络负责将一个域中的图像转换为另一个域中的图像。
你可以选择不同的架构和层数，以适应特定的任务和数据集。
"""

import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,down=True,use_act=True, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,padding_mode="reflect",**kwargs)
            if down
            else nn.ConvTranspose2d(in_channels,out_channels,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)
    

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvBlock(channels,channels,kernel_size=3,padding=1),
            ConvBlock(channels,channels,use_act=False,kernel_size=3,padding=1)
        )

    def forward(self,x):
        return x + self.block(x)
    
class Generator(nn.Module):
    def __init__(self,img_channels,num_features=64,num_residuals=9):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels,num_features,
                      kernel_size=7,stride=1,padding=3,padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features,num_features*2,kernel_size=3,stride=2,padding=1),
                ConvBlock(num_features*2,num_features*4,kernel_size=3,stride=2,padding=1)
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4,num_features*2,
                          down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
                ConvBlock(num_features*2,num_features,
                          down=False,kernel_size=3,stride=2,padding=1,output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features,img_channels,kernel_size=7,stride=1,padding=3,padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))

                
def test():
    in_channels = 3
    img_size = 256
    x = torch.randn((2,in_channels,img_size,img_size))
    gen = Generator(in_channels,9)
    print(gen(x).shape)

if __name__ == '__main__':
    test()