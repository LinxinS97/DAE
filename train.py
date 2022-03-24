import argparse
import glob
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import DAE
from load_cifar10 import MyDataset, train_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--channel", type=int, default=3)
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--gpu", type=int, default=0, help="id of gpu")
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}')

    # DATA
    train_list = glob.glob('./dataset/TRAIN/*/*.png')
    train_dataset = MyDataset(train_list, transform=train_transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)

    print('num_of_train: ', len(train_dataset))

    # MODEL
    model = DAE(args.channel).to(device)

    # LOSS
    loss_fn = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    # TRAIN
    l = len(train_loader)
    losslist = list()
    epochloss = 0
    running_loss = 0

    for epoch in range(1, args.epochs + 1):

        print('Epoch: ', epoch)
        for clean, noise, label in tqdm(train_loader):
            clean = clean.view(-1, args.channel, args.img_size, args.img_size).type(torch.FloatTensor)
            noise = noise.view(-1, args.channel, args.img_size, args.img_size).type(torch.FloatTensor)
            noise, clean = noise.to(device), clean.to(device)

            output = model(noise)
            loss = loss_fn(output, clean)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epochloss += loss.item()
        losslist.append(running_loss / l)
        running_loss = 0
        print('======> epoch: {}/{}, Loss:{}'.format(epoch, args.epochs, loss.item()))
        torch.save(model.state_dict(), 'model/DAE_params_epoch{}.pth'.format(epoch + 1))
