"""
@File : test_mlp.py
@Author : CodeCat
@Time : 2021/7/17 下午4:37
"""
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import save_image

from models.mlp_autoencoder import MLP_AutoEncoder


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)
        ]
    )

    dataset = MNIST('./data', transform=img_transform, train=False, download=True)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False
    )

    model = MLP_AutoEncoder().to(device)

    weight_dict = torch.load('./weights/mlp_autoencoder.pth', map_location=device)
    model.load_state_dict(weight_dict)

    model.eval()
    with torch.no_grad():
        for idx, data in enumerate(dataloader):
            img, _ = data
            save_image(img, './pic/orig.png')
            img = img.view(img.size(0), -1)
            img = Variable(img).to(device)
            output = model(img)
            pic = to_img(output.cpu())
            save_image(pic, './pic/mlp_ae.png')
            break


if __name__ == '__main__':
    main()