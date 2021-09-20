import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def concat_input(in_type, patch_image, patch_grid):
    ''' Returns the defined feature (eventually concatenated).'''

    if in_type == 'rgb':
        patch_in = patch_image
    elif in_type == 'xy':
        patch_in = patch_grid
    else:
        patch_in = torch.cat((patch_image, patch_grid.float()), dim=1)
    return patch_in


def getmodel(setting,  in_chn=5, out_chn=22):
    ''' Returns a model.

    The model can be a CNN, Fully Connected Net, Densenet or U-Net, depending ob the
    setting['name'] that has been chosen.

    Args:
        setting: A dict including the setting for the data, model and training.

    Returns:
        A Model.
    '''

    if setting['name'] == "CNN_Net":
        model = CNN_Net(in_chn, out_chn, setting['kernel_size'], setting['width'],  setting['depth'], setting['input']).to(setting['dev'])
        
    if setting['name'] == "FC_Net":
        model = FC_Net(in_chn, out_chn, setting['width'], setting['depth'], setting['input']).to(setting['dev'])
        
    if setting['name'] == "DenseNet":
        model = DenseNet(in_chn, out_chn, setting['input']).to(setting['dev'])

    if setting['name'] == "UNet":
        model = UNet(in_chn, out_chn, width = setting['width'], in_type=setting['input']).to(setting['dev'])
        
    return model


def conv_relu(width, kernel_size):
    return nn.Sequential(
        nn.Conv2d(width, width, kernel_size=kernel_size, padding=kernel_size//2),
        nn.ReLU()
    )


def linear_relu(width):
    return nn.Sequential(
        nn.Linear(width, width),
        nn.ReLU()
    )


class CNN_Net(nn.Module):
    ''' CNN-Network

    Convolutional Neural Network, with variable width and depth.
    '''


    def __init__(self, in_chn, out_chn, kernel_size, width, depth, in_type):
        ''' Initializes the Module CNN_Net.

        Args:
            in_chn: 
                number of input channel.
            out_chn: 
                number of output channel.
            kernel_size: 
                convolutional kernel size.
            width: 
                width of convolutional layer.
            depth: 
                depth of convolutional layer.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''
        super(CNN_Net, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.in_type = in_type

        assert (kernel_size % 2) == 1

        conv_blocks = [conv_relu(width, kernel_size) 
                       for i in range(depth)]
        
        self.model = nn.Sequential(
            nn.Conv2d(in_chn, width, kernel_size=kernel_size, padding=kernel_size//2),
            nn.LeakyReLU(),
            *conv_blocks,
            nn.Conv2d(width, out_chn, kernel_size=kernel_size, padding=kernel_size//2)
            )

    def forward(self, image, grid):
        ''' Forward Path of the Module CNN_Net.

        Args:
            image: 
                the rgb input image.
            grid: 
                the spacial or semantic features.
        '''
        patch_in = concat_input(self.in_type, image, grid)
        x = self.model(patch_in)
        return x  


class FC_Net(nn.Module):
    ''' FC-Network

    Fully Connected Neural Network, with variable width and depth.
    '''


    def __init__(self, in_chn, out_chn, width, depth, in_type):
        ''' Initializes the Module FC_Net.

        Args:
            in_chn: 
                number of input channel.
            out_chn: 
                number of output channel.
            width: 
                width of Linear layer.
            depth: 
                depth of Linear layer.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''

        super(FC_Net, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.in_type = in_type

        conv_blocks = [linear_relu(width) for i in range(depth)]
        
        self.model = nn.Sequential(
            nn.Linear(in_chn, width),
            nn.ReLU(),
            *conv_blocks,
            nn.Linear(width, out_chn),
            nn.ReLU()
            )

    def forward(self, image, grid):
        patch_in = concat_input(self.in_type, image, grid)
        x = self.model(patch_in)
        return x  


class DenseNet(nn.Module):
    ''' Densenet-Network'''


    def __init__(self, in_chn, out_chn, in_type):
        ''' Initializes the Module Densenet.

        Args:
            in_chn: 
                number of input channel.
            out_chn: 
                number of output channel.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''
        super(DenseNet, self).__init__()

        self.in_chn = in_chn
        self.out_chn = out_chn
        self.in_type = in_type

        kernel_size = 7
        kernel_size_small = 3
        padding = kernel_size//2

        self.conv0 = nn.Conv2d(in_chn, 16, kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(16*1 + in_chn, 16, kernel_size_small, padding=1)
        self.conv2 = nn.Conv2d(16*2 + in_chn, 16, kernel_size_small, padding=1)
        self.conv3 = nn.Conv2d(16*3 + in_chn, 16, kernel_size_small, padding=1)
        self.conv4 = nn.Conv2d(16*4 + in_chn, out_chn, kernel_size_small, padding=1)

    def forward(self, patch_image, patch_grid):
        patch_in = concat_input(self.in_type, patch_image, patch_grid)
        x = (patch_in)
        x = torch.cat((F.relu(self.conv0(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv1(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv2(x)), x), dim=1)
        x = torch.cat((F.relu(self.conv3(x)), x), dim=1)
        x = self.conv4(x)
        return x


class UNet(nn.Module):
    ''' U-Net-Network'''


    def __init__(self, n_channels, n_classes, width=64, bilinear=True, in_type="rgb"):
        ''' Initializes the Module UNet.

        Args:
            n_channels: 
                number of input channel.
            n_classes: 
                number of output channel.
            width: 
                width of Downscaling and Upscaling layers.
            in_type: 
                can become: 'rgb', 'xy' or 'rgbxy'. Decides if the network uses 
                the plain image data, the plain feature or both concatenated as input.
        '''

        super(UNet, self).__init__()

        n_classes = n_classes

        self.in_chn = n_channels
        self.out_chn = n_classes
        self.in_type = in_type

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        width = width

        self.inc = DoubleConv(n_channels, width)
        self.down1 = Down(width, width*2)
        self.down2 = Down(width*2, width*4)
        self.down3 = Down(width*4, width*8)
        self.down4 = Down(width*8, width*8)

        self.up1 = Up(width*16, width*4, bilinear)
        self.up2 = Up(width*8, width*2, bilinear)
        self.up3 = Up(width*4, width, bilinear)
        self.up4 = Up(width*2, width, bilinear)
        self.outc = OutConv(width, n_classes)

    def forward(self, patch_image, patch_grid):
        x = concat_input(self.in_type, patch_image, patch_grid)

        x1 = self.inc(x)
        x2 = self.down1(x1) # Out:128
        x3 = self.down2(x2) # Out:256
        x4 = self.down3(x3) # Out:512
        x5 = self.down4(x4) # Out:512

        x = self.up1(x5, x4)    # In: 512,512   Out: 256
        x = self.up2(x, x3)     # In: 256,256   Out: 128
        x = self.up3(x, x2)     # In: 128,128   Out: 64
        x = self.up4(x, x1)     # In: 64,64     Out: 64

        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""


    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""


    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):


    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
