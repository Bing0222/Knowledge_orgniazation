"""
Discriminator and Generator implementation from DCGAN paper
"""

# Biuild a DCGAN model

import torch
import torch.nn as nn

# Discriminator

class Discriminator(nn.Module):
    """
    判别器负责对图像进行分类，判断其是真实图像还是生成图像。
    通常，判别器由卷积层、批归一化层和激活函数（如LeakyReLU）组成。
    """
    def __init__(self,channels_img,features_d):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            # sigmoid output
            nn.Sigmoid(),
        )
        
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
            return nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                ),
                # nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
            )
        
    def forward(self,x):
            return self.disc(x)


class Generator(nn.Module):
    """
    生成器负责将随机噪声向量转换为逼真的图像。
    通常，生成器由转置卷积层、批归一化层和激活函数（如ReLU）组成。
    最终输出的图像形状应与真实图像一致。
    """
    def __init__(self,channels_noise,channels_img,features_g):
        super(Generator,self).__init__()
        self.net = nn.Sequential(
            # input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # output: N x channels_img x 64 x 64
            nn.Tanh(),
        )
        
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
                nn.ConvTranspose2d(
                    in_channels, out_channels, kernel_size, stride, padding, bias=False
                ),
                # nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        
    def forward(self,x):
        return self.net(x)


# 初始化生成器和判别器
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# Test
def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("All tests passed")


if __name__ == '__main__':
    test()