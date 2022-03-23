from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

BATCH_SIZE = 32

label_name = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]

label_dict = {}

for idx, name in enumerate(label_name):
    label_dict[name] = idx


def default_loader(path):
    return Image.open(path).convert("RGB")


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
    # transforms.Resize((28, 28)),
    transforms.ToTensor()
])


class MyDataset(Dataset):
    def __init__(self,
                 im_list,
                 transform=None,
                 loader=default_loader):
        super(MyDataset, self).__init__()
        imgs = []

        for clean_item, noise_item in im_list:
            # linux
            label_name = clean_item.split('/')[-2]

            # windows
            # im_label_name = im_item.split('\\')[-2]

            imgs.append([clean_item, noise_item, label_dict[label_name]])

        self.imgs = imgs
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        clean_path, noise_path, im_label = self.imgs[index]

        clean_data = self.loader(clean_path)
        noise_data = self.loader(noise_path)

        if self.transform is not None:
            clean_data = self.transform(clean_data)
            noise_data = self.transform(noise_data)

        return clean_data, noise_data, im_label

    def __len__(self):
        return len(self.imgs)

