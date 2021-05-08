import torch
import torch.nn as nn
import torch.nn.functional as F
from math import pi, exp

class LoG(nn.Module):
    def __init__(self, kernel_size=5, sigma=0.65, in_channels=1):
        super(LoG, self).__init__()

        if kernel_size % 2 == 0:
            print("ERROR : Kernel size mag niet even zijn!")
            exit(1)
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.in_channels = in_channels

        kernel = torch.zeros((self.in_channels, 1, self.kernel_size, self.kernel_size), dtype=torch.float, requires_grad=False)

        for x in range(self.kernel_size):
            for y in range(self.kernel_size):
                xv = x - kernel_size//2
                yv = y - kernel_size//2
                kernel[:, :, y, x] = ((-1)/(pi*(self.sigma**4))) * (1-((xv**2 + yv**2)/(2*self.sigma**2))) * exp(-(xv**2 + yv**2)/(2*self.sigma**2))
        
        self.register_buffer('weight', kernel)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=1, padding=self.kernel_size//2, groups=self.in_channels)

class GANLossDisc(nn.Module):
    def __init__(self):
        super(GANLossDisc, self).__init__()

    def forward(self, d, dg):
        """d = real img
           dg = gen img"""
        return (0.5*((d-1)**2) + 0.5*(dg**2))

class GANLossGen(nn.Module):
    def __init__(self):
        super(GANLossGen, self).__init__()
    
    def forward(self, dg):
        return 0.5*((dg-1)**2)