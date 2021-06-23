import torch
import torch.nn as nn
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

class Loss(nn.MSELoss):

    def __init__(self, weight=None):
        super(Loss, self).__init__()        
        #self.bceloss = nn.BCEWithLogitsLoss(pos_weight = weight)
    
    def forward(self, output, target, mask):
        loss = ((output-target)**2) * mask[:,None,:,:].expand_as(output)
        loss = loss.mean(dim=3).mean(dim=2).mean(dim=1).mean(dim=0)
        
        return loss

class LabelSmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.1):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets: torch.Tensor, n_classes: int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                                    .fill_(smoothing / (n_classes - 1)) \
                                    .scatter_(1, targets.data.unsqueeze(1), 1. - smoothing)
        return targets


    def forward(self, inputs, targets):
        targets = LabelSmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
                self.smoothing)
        lsm = F.log_softmax(inputs, -1)
        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)
        loss = -(targets * lsm).sum(-1)
        if self.reduction == 'sum':
            loss = loss.sum()

        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss

class Custom_CrossEntropyLoss(LabelSmoothCrossEntropyLoss):
    def __init__(self, weight = None):
        super(Custom_CrossEntropyLoss,self).__init__()

        self.weight = weight

    def forward(self,output, label):
        celoss = LabelSmoothCrossEntropyLoss(weight = self.weight, reduction = 'mean')
        loss = 0
        for i in range(9):
            for j in range(9):
                loss += celoss(output[:,:,i,j], label[:,i,j])
        return loss/(9*9)
