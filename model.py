import torch.nn as nn

def conv_block(in_c, out_c, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, stride=stride),
        nn.BatchNorm2d(out_c),
        nn.ReLU()
    )

def linear_block(in_dim, out_dim, **kwargs):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, **kwargs),
        nn.ReLU()
    )

class DCNN(nn.Module):
    def __init__(self, c):
        super(DCNN, self).__init__()
        # self.encoder = nn.Sequential(
        #     conv_block(c, 16),
        #     conv_block(16, 32),
        #     conv_block(32, 64),
        #     nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
        # )
        #
        # self.decoder = nn.Sequential(
        #     conv_block(64, 64),
        #     conv_block(64, 32),
        #     conv_block(32, 16),
        #     nn.Conv2d(16, c, kernel_size=(3, 3), padding=1),
        # )
        self.encoder = nn.Sequential(
            conv_block(c, 2),
            conv_block(2, 2),
            nn.Conv2d(2, 1, kernel_size=(3, 3), padding=1),
        )

        self.decoder = nn.Sequential(
            conv_block(1, 2),
            conv_block(2, 2),
            nn.Conv2d(2, c, kernel_size=(3, 3), padding=1),
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class DAE(nn.Module):
    def __init__(self, c, img_size):
        super(DAE, self).__init__()
        self.encoder = nn.Sequential(
            linear_block(c*img_size**2, 512),
            linear_block(512, 256),
            linear_block(256, 128),
        )
        self.decoder = nn.Sequential(
            linear_block(128, 256),
            linear_block(256, 512),
            nn.Linear(512, c*img_size**2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x