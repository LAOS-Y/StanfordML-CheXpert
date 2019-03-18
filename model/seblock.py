import torch.nn as nn

class SELayer(nn.Module):
    def __init__(self, channel, ratio=4, initer=None):
        super(SELayer, self).__init__()
        self.fc1 = nn.Linear(in_features=channel, out_features=channel // ratio)
        self.fc2 = nn.Linear(in_features=channel // ratio, out_features=channel)
        
        self.channel = channel
        self.initer = initer
        
        if self.initer is not None:
            self.initer(self.fc1.weight)
            self.initer(self.fc2.weight)
        
    def forward(self, x):
        out = nn.AdaptiveAvgPool2d(1)(x)
        out = self.fc1(out.view((-1, self.channel)))
        out = nn.ReLU()(out)
        out = self.fc2(out)
        out = nn.Sigmoid()(out).view(-1, self.channel, 1, 1)
        
        return out * x
    
class SEBlock(nn.Module):
    def __init__(self, ori_block, channel, ratio=4, initer=None):
        super(SEBlock, self).__init__()
        self.ori_block = ori_block
        self.se_layer = SELayer(channel, ratio, initer)
            
    def forward(self, x):
        out = self.ori_block(x)
        out = self.se_layer(out)
        
        return out