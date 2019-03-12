import torch
import torch.nn as nn

class CAM(nn.Module):
    def __init__(self):
        super(CAM, self).__init__()

    def forward(self, feature_map, classifier_weight):
        N, D, H, W = feature_map.shape
        
        feature_map = feature_map.view(N, D, H * W)
        feature_map.transpose_(2, 1)
        feature_map = torch.matmul(feature_map, classifier_weight.transpose(1, 0))
        feature_map.transpose_(2, 1)
        return feature_map.view(N, -1, H, W)
