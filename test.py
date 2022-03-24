import glob
import cv2
import torch
from torch.utils.data import DataLoader
from load_cifar10 import MyDataset, test_transform
from model import DAE


def sample_batch(loader):
    while True:
        for batch in loader:
            yield batch

path = './res/'

channel = 3
img_size = 32
test_list = glob.glob('./dataset/TEST/*/*.png')
test_dataset = MyDataset(test_list, transform=test_transform)


test_loader = DataLoader(dataset=test_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=4)
test_loader = sample_batch(test_loader)

model = DAE(channel)
model.load_state_dict(torch.load('./model/DAE_params_epoch200.pth'))
model.eval()

print('num_of_test: ', len(test_dataset))

for idx in range(10):
    clean, noise, label = next(test_loader)
    clean = clean.view(-1, channel, img_size, img_size).type(torch.FloatTensor)
    noise = noise.view(-1, channel, img_size, img_size).type(torch.FloatTensor)

    output = model(noise)

    output = output.view(3, 32, 32)
    output = output.permute(1, 2, 0)
    output = output.detach().cpu().numpy()

    noise = noise.view(3, 32, 32)
    noise = noise.permute(1, 2, 0)
    noise = noise.detach().cpu().numpy()

    clean = clean.view(3, 32, 32)
    clean = clean.permute(1, 2, 0)
    clean = clean.detach().cpu().numpy()

    # imshow在wsl上会出问题，所以这里用了保存文件的方式
    cv2.imwrite('./res/{}_clean.png'.format(idx), clean*255)
    cv2.imwrite('./res/{}_noise.png'.format(idx), noise*255)
    cv2.imwrite('./res/{}_output.png'.format(idx), output*255)

    # cv2.imshow(clean*255)
    # cv2.imshow(noise*255)
    # cv2.imshow(output*255)