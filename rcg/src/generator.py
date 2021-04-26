import torch
import torch.nn as nn
import torchvision.models as models

class GeneratorNetwork(nn.Module):
    def __init__(self, args):

        super(GeneratorNetwork, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(args.img_ch * args.input_length, 128, 7, 1, padding=3),
            ConvLayer(128, 128, 3, 2),
            ConvLayer(128, 256, 3, 2),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            ResLayer(256, 256, 3, 1),
            TransposeConvLayer(256, 128, 3, 2),
            TransposeConvLayer(128, 256, 3, 2),
            nn.Conv2d(256, args.img_ch, 7, 1, padding=3),
            nn.Sigmoid())

    def forward(self, input):
        return self.layers(input)

class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):

        super(ConvLayer, self).__init__()

        self.pad = nn.ZeroPad2d(kernel_size//2)

        self.conv_layer = nn.Conv2d(in_ch,
                                    out_ch, 
                                    kernel_size, 
                                    stride) #convolutional layer
        
        self.instanceBN = nn.InstanceNorm2d(out_ch) #instance normalisation

        self.activation = nn.ReLU() #relu activation
    
    def forward(self, x):
        x = self.pad(x)
        x = self.conv_layer(x)
        x = self.instanceBN(x)
        x = self.activation(x)
        return x
        

class ResLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):

        super(ResLayer, self).__init__()

        self.conv_layer1 = ConvLayer(in_ch, out_ch, kernel_size, stride) #convlayer1

        self.conv_layer2 = ConvLayer(in_ch, out_ch, kernel_size, stride) #convlayer2

    def forward(self, x):
        y = self.conv_layer1(x)
        return self.conv_layer2(y) + x

class TransposeConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):

        super(TransposeConvLayer, self).__init__()
        
        self.transpose_conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride, padding=1, output_padding=1)
        
        self.instanceBN = nn.InstanceNorm2d(out_ch) #batch normalisation

        self.activation = nn.ReLU() #relu activation
    
    def forward(self, x):
        x = self.transpose_conv(x)
        x = self.instanceBN(x)
        x = self.activation(x)
        return x



