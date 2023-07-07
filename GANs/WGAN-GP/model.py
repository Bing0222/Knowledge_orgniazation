"""
Discriminator and Generator implementation from DCGAN paper


WGAN-GP（Wasserstein生成对抗网络带有梯度惩罚）是生成对抗网络（GAN）的一种变体，
旨在改善传统GAN存在的训练不稳定和模式崩溃等问题。
WGAN-GP引入了Wasserstein距离的概念，以提供更好的梯度特性和生成样本质量。

以下是WGAN-GP的关键概念和步骤：

Wasserstein距离：
Wasserstein距离（也称为地球距离）是一种衡量两个概率分布之间的差异的度量。
在WGAN-GP中，将Wasserstein距离应用于生成器和判别器之间的差异，以取代传统GAN中使用的交叉熵损失。Wasserstein距离能够提供更平滑的梯度，从而帮助生成器和判别器的训练更加稳定。

梯度惩罚：
为了约束判别器的参数空间，WGAN-GP引入了梯度惩罚。
在计算判别器损失时，除了计算真实样本和生成样本在判别器中的输出之外，还在真实样本和生成样本之间进行随机采样，并在梯度上施加一个惩罚项。这个惩罚项促使判别器的梯度范围受限，从而提高训练的稳定性。

判别器和生成器的训练：
WGAN-GP的训练过程分为两个阶段：首先是训练判别器，然后是训练生成器。

训练判别器：判别器的目标是最大化真实样本的输出（Wasserstein距离中的正项）
并最小化生成样本的输出（Wasserstein距离中的负项）。
在计算损失时，通过减去两个输出的平均值来得到判别器的损失。
然后，使用反向传播算法更新判别器的参数。
训练生成器：生成器的目标是最大化生成样本在判别器中的输出。
通过计算生成样本在判别器中的输出，并取其平均值的负值作为生成器的损失。然后，使用反向传播算法更新生成器的参数。
梯度截断：
为了限制判别器的参数范围，可以对判别器的参数进行截断操作。
在训练过程中，可以通过将判别器的参数限制在一个预定范围内，如[-0.01, 0.01]，以帮助稳定训练。
"""

import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels_img,features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input is n x c x 64 x 64
            nn.Conv2d(channels_img, features_d, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x):
        return self.disc(x)
    

class Generator(nn.Module):
    def __init__(self,channels_noise,channels_img,features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input is Z, going into a convolution
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output is n x c x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size, stride, padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.net(x)
    
def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Discriminator(in_channels, 8)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    print("All tests passed!")

if __name__ == '__main__':
    test()