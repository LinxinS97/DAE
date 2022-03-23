import glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DAE
from load_cifar10 import MyDataset, train_transform

epochs = 200
batch_size = 128
lr = 0.01
weight_decay = 1e-5
channel = 3
img_size = 32

# DATA
train_list = glob.glob('./dataset/TRAIN/*/*.png')
train_dataset = MyDataset(train_list, transform=train_transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

print('num_of_train: ', len(train_dataset))

# MODEL
model = DAE(channel).cuda()

# LOSS
loss_fn = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

# TRAIN
l = len(train_loader)
losslist = list()
epochloss = 0
running_loss = 0

for epoch in range(1, epochs + 1):

    print('Epoch: ', epoch)
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
    print('======> epoch: {}/{}, Loss:{}'.format(epoch, epochs, loss.item()))
    torch.save(model.state_dict(), 'model/DAE_params_epoch{}.pth'.format(epoch + 1))
