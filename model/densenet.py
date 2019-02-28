import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121

class Densenet121(nn.Module):
    def __init__(self, num_class=14, pretrained=True):
        super(Densenet121, self).__init__()
        model = densenet121(pretrained)
        self.features = model.features
        self.classifier = nn.Linear(in_features=1024, out_features=num_class)
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        #out = nn.AdaptiveAvgPool2d(1)(out).view(features.size(0), -1)
        out = F.avg_pool2d(out, kernel_size=7, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        #out = nn.Sigmoid()(out)
        return out