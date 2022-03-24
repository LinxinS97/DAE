import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


label_name = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")

def add_noise(img, noise_type='gaussian'):
    if noise_type == 'gaussian':
        mean = 0
        sigma = 0.2
        normal = torch.distributions.Normal(mean, sigma)

        noise = normal.sample(sample_shape=torch.Size(img.shape))
        img = img + noise
        return img

    if noise_type == 'speckle':
        noise = torch.randn(img.shape)
        img = img + img * noise
        return img


train_transform = transforms.Compose([
    # transforms.RandomResizedCrop((28, 28)),
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(90),
    # transforms.RandomGrayscale(0.1),
    # transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self,
                 im_list,
                 transform=None,
                 loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for clean_item in im_list:
            # linux
            label_name = clean_item.split('/')[-2]

            # windows
            # im_label_name = im_item.split('\\')[-2]

            imgs.append([clean_item, label_dict[label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        clean_path, im_label = self.imgs[index]
        clean_data = self.loader(clean_path)

        if self.transform is not None:
            clean_data = self.transform(clean_data)

        # if np.random.randint(2, size=1) == 0:
        #     noise_data = add_noise(clean_data, noise_type='gaussian')
        # else:
        noise_data = add_noise(clean_data, noise_type='gaussian')

        return clean_data, noise_data, im_label

    def __len__(self):
        return len(self.imgs)

