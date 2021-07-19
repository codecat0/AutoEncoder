"""
@File : test.py
@Author : CodeCat
@Time : 2021/7/17 下午4:28
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from models.vae import VAE


def to_img(x):
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    dataset = MNIST('./data', transform=img_transform, train=False, download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False
    )

    model = VAE().to(device)

    weight_dict = torch.load('./weights/vae_autoencoder.pth', map_location=device)
    model.load_state_dict(weight_dict)

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            img, _ = data
            img = img.view(img.size(0), -1)
            img = Variable(img).to(device)
            recon_batch, mu, logvar = model(img)
            pic = to_img(recon_batch.cpu())
            save_image(pic, './pic/vae.png')
            break


if __name__ == '__main__':
    main()