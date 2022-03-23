import glob
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DAE
from load_cifar10 import MyDataset, train_transform

epochs = 30
batch_size = 32
channel = 3
img_size = 32
lr = 0.001
weight_decay = 1e-5

# DATA
train_clean_list = glob.glob('./dataset/TRAIN/*/*.png')
train_noise_list = glob.glob('./dataset/TRAIN_NOISE/*/*.png')
# test_clean_list = glob.glob('./dataset/TEST/*/*.png')
# test_noise_list = glob.glob('./dataset/TEST/*/*.png')

train_dataset = MyDataset(np.array(list(zip(train_clean_list, train_noise_list))), transform=train_transform)
# test_dataset = MyDataset(np.array(list(zip(test_clean_list, test_noise_list))), transform=test_transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)
# test_loader = DataLoader(dataset=test_dataset,
#                          batch_size=batch_size,
#                          shuffle=False,
#                          num_workers=4)

print("num_of_train: ", len(train_dataset))
# print("num_of_test: ", len(test_dataset))

# MODEL
model = DAE(channel).cuda()

# LOSS
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# TRAIN
l = len(train_loader)
losslist = list()
epochloss = 0
running_loss = 0

for epoch in range(epochs):

    print("Entering Epoch: ", epoch)
    for clean, noise, label in tqdm(train_loader):
        clean = clean.view(-1, channel, img_size, img_size).type(torch.FloatTensor)
        noise = noise.view(-1, channel, img_size, img_size).type(torch.FloatTensor)
        noise, clean = noise.cuda(), clean.cuda()

        output = model(noise)
        loss = loss_fn(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        epochloss += loss.item()
    losslist.append(running_loss / l)
    running_loss = 0
    print("======> epoch: {}/{}, Loss:{}".format(epoch, epochs, loss.item()))
    torch.save(model.state_dict(), "model/DAE_params_epoch{}.pth".format(epoch + 1))
