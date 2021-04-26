import torch
import torch.nn as nn
import torchvision.models as models

class DiscriminatorNetwork(nn.Module):
    def __init__(self, args, frame):

        super(DiscriminatorNetwork, self).__init__()

        if frame:
            input_channels = args.img_ch
        else:
            input_channels = args.img_ch * (args.input_length + 1)
        
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, 2, padding=1),
            nn.LeakyReLU(negative_slope=0.2),
            ConvLayer(64, 128, 4, 2),
            ConvLayer(128, 256, 4, 2),
            ConvLayer(256, 512, 4, 1),
            nn.ZeroPad2d((2, 1, 2, 1)),
            nn.Conv2d(512, 1, 4, 1))
    
    def forward(self, input):
        return self.layers(input)

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):

        super(ConvLayer, self).__init__()

        if stride == 2:
            self.pad = nn.ZeroPad2d(1)
        elif stride == 1:
            self.pad = nn.ZeroPad2d((2, 1, 2, 1))

        self.conv_layer = nn.Conv2d(in_ch, out_ch, 
                                    kernel_size, 
                                    stride) #convolutional layer
        
        self.instanceBN = nn.InstanceNorm2d(out_ch) #instance normalisation

        self.activation = nn.LeakyReLU(negative_slope=0.2) #leaky relu activation; negative slope=0.2 (paper)
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.instanceBN(x)
        x = self.activation(x)
        return x