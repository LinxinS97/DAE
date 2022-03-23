import torch.nn as nn

def conv_block(in_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

class DAE(nn.Module):
    def __init__(self, c):
        super(DAE, self).__init__()

        self.encoder = nn.Sequential(
            conv_block(c, 64),
            conv_block(64, 64),
            conv_block(64, 128),
            nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
        )

        self.decoder = nn.Sequential(
            conv_block(128, 128),
            conv_block(128, 64),
            conv_block(64, 64),
            nn.Conv2d(64, c, kernel_size=(3, 3), padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x