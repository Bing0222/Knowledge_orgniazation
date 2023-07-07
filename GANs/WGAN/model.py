"""
当谈论WGAN（Wasserstein生成对抗网络）时，
通常是指WGAN的原始版本，也称为WGAN-GP（Wasserstein生成对抗网络带有梯度惩罚）


问题背景：
WGAN-GP是生成对抗网络（GAN）的一种变体，旨在改善传统GAN存在的训练不稳定和模式崩溃等问题。
它引入了一个Wasserstein距离的概念，以帮助更好地估计生成器和判别器之间的差异。

网络结构：
WGAN-GP与传统GAN类似，由生成器（Generator）和判别器（Discriminator）组成。
生成器尝试生成逼真的样本，而判别器则努力区分真实样本和生成样本。

损失函数：
WGAN-GP使用Wasserstein距离作为判别器的损失函数，而不是传统GAN中使用的交叉熵损失。
Wasserstein距离（也称为地球距离）测量了两个分布之间的差异，具有更好的梯度特性。

训练过程：
WGAN-GP的训练过程分为多个步骤：
a. 从数据集中抽样一批真实样本。
b. 生成器接收一个随机噪声向量作为输入，并生成一批合成样本。
c. 计算真实样本和合成样本之间的Wasserstein距离。这可以通过将真实样本传递给判别器，
并计算其输出与合成样本传递给判别器后的输出之间的差异来实现。
d. 计算梯度惩罚项。这是WGAN-GP的关键步骤，通过在真实样本和合成样本之间随机采样，
并在梯度上施加惩罚，以确保判别器的梯度范围受限。
e. 更新判别器的权重。通过最小化Wasserstein距离和梯度惩罚项的组合损失来调整判别器的参数。
f. 更新生成器的权重。通过最大化生成器输出被判别器误分类的程度来调整生成器的参数。
g. 重复步骤a至f，直到达到预定的训练迭代次数或达到期望的生成样本质量。

超参数调整：
WGAN-GP有一些超参数需要调整，如学习率、批量大小、梯度惩罚系数等。
这些超参数的选择对于训练稳定性和生成样本的质量至关重要。
"""

import torch
import torch.nn as nn

# Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input shape: (N, in_channels, H, W)
            nn.Conv2d(
                in_channels, features, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self._block(features, features * 2, 4, 2, 1),
            self._block(features * 2, features * 4, 4, 2, 1),
            self._block(features * 4, features * 8, 4, 2, 1),
            # output shape: (N, features * 8, 4, 4)
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=2, padding=0),
            # output shape: (N, 1, 1, 1)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels,
                        kernel_size, 
                        stride, padding, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            )
    
    def forward(self, x):
        return self.disc(x)

# Generator

class Generator(nn.Module):
    def __init__(self, z_dim, channels=1, features=64):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # input shape: (N, z_dim)
            self._block(z_dim, features * 8, 4, 1, 0),
            self._block(features * 8, features * 4, 4, 2, 1),
            self._block(features * 4, features * 2, 4, 2, 1),
            self._block(features * 2, features, 4, 2, 1),
            # output shape: (N, channels, 32, 32)
            nn.ConvTranspose2d(features, channels, kernel_size=4, stride=2, padding=1),
            # output shape: (N, channels, 64, 64)
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, 
                               out_channels,
                               kernel_size,
                               stride, padding, bias=False),
                               nn.BatchNorm2d(out_channels),
                               nn.ReLU(),
                               )

    def forward(self, x):
        return self.gen(x)
    

# initialize_weights
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)

def test():
    N,in_channels,H,W = 8,3,64,64
    noise_dim = 100
    x = torch.randn((N,in_channels,H,W))
    disc = Discriminator(in_channels,8)
    assert disc(x).shape == (N,1,1,1),  "Discriminator test failed"
    gen = Generator(noise_dim,in_channels,8)
    z = torch.randn((N,noise_dim,1,1))
    assert gen(z).shape == (N,in_channels,H,W), "Generator test failed"
    print("All tests passed!")


if __name__ == '__main__':
    test()
