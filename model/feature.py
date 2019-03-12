import torch
import torch.nn as nn
from torchvision.models import resnet50, densenet121

class FeatureResnet50(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureResnet50, self).__init__()
        model = resnet50(pretrained)
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.relu = model.relu
        self.maxpool = model.maxpool
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class FeatureDensenet121(nn.Module):
    def __init__(self, pretrained=True):
        super(FeatureDensenet121, self).__init__()
        model = densenet121(pretrained)
        self.features = model.features
    
    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        return out

class Feature(nn.Module):
    def __init__(self, backbone, pretrained=True):
        super(Feature, self).__init__()
        features = {"Resnet50":FeatureResnet50,
                    "Densenet121":FeatureDensenet121}
        self.feature = features[backbone](pretrained)

    def forward(self, x):
        return self.feature(x)
