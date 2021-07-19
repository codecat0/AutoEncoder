"""
@File : train.py
@Author : CodeCat
@Time : 2021/7/17 上午11:28
"""
import math
import argparse

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST

from models.vae import VAE



def loss_function(recon_x, x, mu, logvar):
    """
    :param recon_x: 生成的图像
    :param x: 原始图像
    :param mu: 均值
    :param logvar: log方差
    :return:
    """
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # KL = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = MNIST('./data', transform=img_transform, download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=opt.batch_size,
        shuffle=True
    )

    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - 0.01) + 0.01
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(opt.epochs):
        model.train()
        epoch_loss = 0
        for idx, data in enumerate(dataloader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).to(device)
            recon_batch, mu, logvar = model(img)
            loss = loss_function(recon_batch, img, mu, logvar)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        print("[Epoch {}/{}], epoch_loss is {:.8f}".format(epoch+1, opt.epochs, epoch_loss/len(dataset)))

    torch.save(model.state_dict(), './weights/vae_autoencoder.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=100)
    opt = parser.parse_args()
    print(opt)
    main(opt)

