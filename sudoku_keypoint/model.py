import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
                    nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
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
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

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



class Keypoint_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=82+1, bilinear=True):
        super(Keypoint_Net, self).__init__()
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
        

        self.keyc = nn.Conv2d(64,4,kernel_size = 1)
        self.dp = nn.Dropout(0.5)       

    def forward(self, input):

        x1 = self.inc(input)
        x2 = self.down1(x1)        
        x3 = self.down2(x2)        
        x4 = self.down3(x3)        
        x5 = self.down4(x4)   
        x = self.up1(x5, x4)        
        x = self.up2(x, x3)   
        x = self.up3(x, x2)   
        x = self.up4(x, x1)        
        x1 = self.outc(x)
        logits3 = self.keyc(x)
        #logits3 = self.dp(x[:,:1,:,:])
        
        logits2 = x1[:,1:,:,:]
        logits1 = x1[:,0,:,:]
        
        return logits1, logits2, logits3


if __name__=='__main__':
    #d = torch.randn((2,3,300,300))
    model = Uchan_Net().cuda()
    #a,b = model(d)
    
    #print(a.shape)
    #print(b.shape)

    summary(model, (3,270,270))
