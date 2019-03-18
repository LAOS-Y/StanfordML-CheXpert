import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121
from .seblock import SEBlock

class Densenet121(nn.Module):
    def __init__(self, num_class=14, pretrained=True, initer=None, se_ratio=None):
        super(Densenet121, self).__init__()
        model = densenet121(pretrained)
        self.features = model.features
        
        self.classifier = nn.Linear(in_features=1024, out_features=num_class)
        self.initer = initer
        
        if self.initer is not None:
            self.initer(self.classifier.weight)
            
        self.se_ratio = se_ratio
        if self.se_ratio is not None:
            self._features2SEBlock(self.se_ratio)
    
    def _features2SEBlock(self, ratio):
        self.features.denseblock1 = SEBlock(self.features.denseblock1,
                                            channel=256,
                                            ratio=ratio,
                                            initer=self.initer)
        self.features.denseblock2 = SEBlock(self.features.denseblock2,
                                            channel=512,
                                            ratio=ratio,
                                            initer=self.initer)
        self.features.denseblock3 = SEBlock(self.features.denseblock3,
                                            channel=1024,
                                            ratio=ratio,
                                            initer=self.initer)
        self.features.denseblock4 = SEBlock(self.features.denseblock4,
                                            channel=1024,
                                            ratio=ratio,
                                            initer=self.initer)
        
    
    def forward(self, x):
        features = self.features(x)
        out = nn.ReLU(inplace=True)(features)
        out = nn.AdaptiveAvgPool2d(1)(out).view(features.size(0), -1)
        out = self.classifier(out)
        return out

def SEDensenet121(num_class=14, pretrained=True, initer=None, se_ratio=4):
    return Densenet121(num_class, pretrained, initer, se_ratio)