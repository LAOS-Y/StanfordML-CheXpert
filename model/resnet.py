import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from .seblock import SEBlock
from .backbone import Backbone
from collections import OrderedDict

class Resnet50(Backbone):
    def __init__(self, num_class=14, pretrained=True, initer=None, se_ratio=None):
        super(Resnet50, self).__init__()
        model = resnet50(pretrained)

        self.features = nn.Sequential(OrderedDict([('conv1', model.conv1),
                                                   ('bn1', model.bn1),
                                                   ('relu', model.relu),
                                                   ('maxpool', model.maxpool)]))
        
        for i, block in enumerate([model.layer1, model.layer2, model.layer3, model.layer4]):
            self.features.add_module("layer{}".format(i + 1), block)
        
#         self.features = nn.Sequential(model.conv1,
#                                       model.bn1,
#                                       model.relu,
#                                       model.maxpool,
#                                       model.layer1,
#                                       model.layer2,
#                                       model.layer3,
#                                       model.layer4)
        
        self.avgpool = model.avgpool
        self.classifier = nn.Linear(in_features=2048, out_features=num_class)
    
        self.initer = initer

        if self.initer is not None:
            self.initer(self.classifier.weight)
            
        self.se_ratio = se_ratio
        if self.se_ratio is not None:
            self._features2SEBlock(self.se_ratio)

    def _features2SEBlock(self, ratio):
        self.features.layer1 = SEBlock(self.features.layer1,
                                       channel=256,
                                       ratio=ratio,
                                       initer=self.initer)
        self.features.layer2 = SEBlock(self.features.layer2,
                                       channel=512,
                                       ratio=ratio,
                                       initer=self.initer)
        self.features.layer3 = SEBlock(self.features.layer3,
                                       channel=1024,
                                       ratio=ratio,
                                       initer=self.initer)
        self.features.layer4 = SEBlock(self.features.layer4,
                                       channel=2048,
                                       ratio=ratio,
                                       initer=self.initer)
            
    def forward(self, x):
        x = self.features(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x
    
def SEResnet50(num_class=14, pretrained=True, initer=None, se_ratio=4):
    return Resnet50(num_class, pretrained, initer, se_ratio)
