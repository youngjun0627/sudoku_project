import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import timm

class Number_Net(nn.Module):
    def __init__(self):
        super(Number_Net, self).__init__()
        
        base_model = timm.create_model('efficientnet_b2', pretrained = True, in_chans=3, drop_rate = 0.3, drop_path_rate=0.3, num_classes = 10*9*9)
        feature_num = base_model.num_features
        self.extractor = nn.Sequential(*list(base_model.children())[:-2])

        self.conv1 = nn.Conv2d(feature_num,1024,kernel_size = 1)
        self.conv2 = nn.Conv2d(1024,512,kernel_size = 3, padding=1)
        self.conv3 = nn.Conv2d(feature_num+512,10,kernel_size = 1)
        self.dp = nn.Dropout(0.5)       
    def forward(self, input):

        output = self.dp(self.extractor(input))
        x = self.dp(self.conv1(output))
        x = self.dp(self.conv2(x))
        output = self.conv3(torch.cat([x,output], dim=1))
        
        return output

if __name__=='__main__':
    #d = torch.randn((2,3,300,300))
    model = Number_Net().cuda()
    #a,b = model(d)
    
    #print(a.shape)
    #print(b.shape)

    summary(model, (3,270,270))
