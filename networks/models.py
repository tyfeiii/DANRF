""" Full assembly of the parts to form the complete network """

import torch.nn.functional as F
from torch.nn.modules import Sequential
from networks.unet_parts import *


class UNet1(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet1, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)



    def forward(self, x):
        x0 = x
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        logits = x0 - logits


        return logits,x5


if __name__ == '__main__':
    m = UNet1(1,1)
    input1 = torch.randn(21, 1, 64, 64)
    # input2=torch.randn(21,512,8,8)
    (output1,output2) = m(input1)
    print(output1.shape)
    print(output2.shape)
    mse= torch.nn.MSELoss(output2,output2)
    print(mse.shape)
