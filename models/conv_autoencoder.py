"""
@File : conv_autoencoder.py
@Author : CodeCat
@Time : 2021/7/16 下午10:35
"""
from torch import nn


class Conv_AutoEncoder(nn.Module):
    def __init__(self):
        super(Conv_AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            # [b, 16, 10, 10]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 16, 5, 5]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [b, 8, 3, 3]
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # [b, 8, 2, 2]
            nn.MaxPool2d(kernel_size=2, stride=1)
        )
        self.decoder = nn.Sequential(
            # [b, 16, 5, 5]
            nn.ConvTranspose2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            # [b, 8, 15, 15]
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=3, padding=1),
            nn.ReLU(inplace=True),
            # [b, 1, 28, 28]
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=2, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
