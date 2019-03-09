import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
    
class Resnet50(nn.Module):
    def __init__(self, num_class=14, pretrained=True, classifier_init=None):
        super(Resnet50, self).__init__()
        model = resnet50(pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
        self.avgpool = model.avgpool
        self.classifier = nn.Linear(in_features=2048, out_features=num_class)
    
        self.classifier_init = classifier_init

        if self.classifier_init is not None:
            self.classifier_init(self.classifier.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
