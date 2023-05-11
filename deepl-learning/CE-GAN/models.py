import torch.nn as nn
# Generator model
class Generator(nn.Module):
    def __init__(self, noise_size,base_filters=64):
        # Call parent
        super().__init__()
        nf = base_filters
        self.layers = nn.Sequential(
            nn.Conv2d(channel, nf, 4, 2, 1,bias=False), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf*2, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*4, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4, nf*8, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*8, noise_size, 4,bias=False), 
            nn.Sequential(
              nn.BatchNorm2d(noise_size),
              nn.LeakyReLU(0.2, True),
              nn.ConvTranspose2d(noise_size, nf*8, 4, 1, 0,bias=False), 
              nn.BatchNorm2d(nf*8),
              nn.ReLU(True),
              nn.ConvTranspose2d(nf*8, nf*4, 4, 2, 1,bias=False), 
              nn.BatchNorm2d(nf*4),
              nn.ReLU(True),
              nn.ConvTranspose2d(nf*4, nf*2, 4, 2, 1,bias=False), 
              nn.BatchNorm2d(nf*2),
              nn.ReLU(True),
              nn.ConvTranspose2d(nf*2, nf, 4, 2, 1,bias=False), 
              nn.BatchNorm2d(nf),
              nn.ReLU(True),
              nn.ConvTranspose2d(nf, channel, 4, 2, 1,bias=False), 
              nn.Tanh())
        )
    def forward(self, x):
        x = self.layers(x)
        return x
     
class Discriminator(nn.Module):
    def __init__(self, base_filters=64):
        # Call parent
        super().__init__()
        nf = base_filters
        self.layers = nn.Sequential(
            nn.Conv2d(3, nf, 4, 2, 1,bias=False), 
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf, nf*2, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*2, nf*4, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*4, nf*8, 4, 2, 1,bias=False), 
            nn.BatchNorm2d(nf*8),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(nf*8, 1, 4, 1, 0,bias=False), 
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = self.layers(x)
        # Remove dimensions from output
        x = x.view(-1)
        return x