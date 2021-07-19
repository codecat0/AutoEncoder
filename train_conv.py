"""
@File : train.py
@Author : CodeCat
@Time : 2021/7/16 下午10:46
"""
import os
import argparse

import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from models.conv_autoencoder import Conv_AutoEncoder


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ]
    )

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )

    model = Conv_AutoEncoder().to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)

    for epoch in range(opt.epochs):
        epoch_loss = 0
        for idx, data in enumerate(dataloader):
            img, _ = data
            img = Variable(img).to(device)
            output = model(img)
            loss = loss_func(output, img)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print("[Epoch {}/{}], epoch_loss is {:.8f}".format(epoch+1, opt.epochs, epoch_loss/len(dataset)))

    torch.save(model.state_dict(), './weights/conv_autoencoder.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    opt = parser.parse_args()
    print(opt)
    main(opt)

