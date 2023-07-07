"""
Build an Efficient Net
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, 
                 kernel_size, stride=1, padding=0,bias=False):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size, stride=stride, padding=padding,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels,eps=0.001,momentum=0.01)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu6(x, inplace=True)
        
        
class Swish(nn.Module):
    def forward(self, x):
        return x*torch.sigmoid(x)
    


class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride=1, padding=0,bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, 
                              kernel_size, stride=stride, padding=padding,bias=bias,groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,expand_ratio
                 ,kernel_size, stride=1, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.expand_channels = in_channels * expand_ratio
        self.use_residual = in_channels == out_channels and stride == 1

        if expand_ratio != 1:
            self.expand_conv = Conv2d(in_channels, self.expand_channels, kernel_size=1)

        self.dw_conv = DepthwiseConv2d(self.expand_channels, out_channels,
                                        kernel_size, stride=stride,padding=(kernel_size-1)//2)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, int(out_channels*se_ratio), kernel_size=1),
            Swish(),
            nn.Conv2d(int(out_channels*se_ratio), out_channels, kernel_size=1),
            nn.Sigmoid()
            )
        
        self.pw_conv = Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x

        if hasattr(self, 'expand_conv'):
            x = self.expand_conv(x)
        x = self.dw_conv(x)

        if self.se is not None:
            se_weight = self.se(x)
            x = x * se_weight

        x = self.pw_conv(x)

        if self.use_residual:
            x = x + identity

        return x
        


class EfficientNet(nn.Module):
    def __init__(self,num_classes,
                 width_coef=1.0,depth_coef=1.0,
                 resolution_coef=1.0,dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        self.config = [
            # t, c, n, s, se
            [1,  16, 1, 1, 0.25],
            [6,  24, 2, 2, 0.25],
            [6,  40, 2, 2, 0.25],
            [6,  80, 3, 2, 0.25],
            [6, 112, 3, 1, 0.25],
            [6, 192, 4, 2, 0.25],
            [6, 320, 1, 1, 0.25]
        ]
        self.num_classes = num_classes
        self.first_conv = Conv2d(3, int(32*width_coef), kernel_size=3, stride=2, padding=1)
        
        blocks = []
        in_channels = int(32*width_coef)
        for t, c, n, s, se in self.config:
            out_channels = int(c*width_coef)
            layers = [MBConvBlock(in_channels, out_channels,expand_ratio=t,
                                  kernel_size=3, stride=s,se_ratio=se)]
            for i in range(n-1):
                layers.append(MBConvBlock(out_channels, out_channels,
                                          expand_ratio=t,kernel_size=3,se_ratio=se))
                blocks.extend(layers)
                in_channels = out_channels


        self.blocks = nn.Sequential(*blocks)

        self.last_conv = Conv2d(in_channels, int(1280*width_coef), kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(int(1280*width_coef), num_classes)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.blocks(x)
        x = self.last_conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
num_classes = 10 
model = EfficientNet(num_classes)

print(model)