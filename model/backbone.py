import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self, num_class=14, pretrained=True, initer=None, se_ratio=None):
        super(Backbone, self).__init__()

    def _features2SEBlock(self, ratio):
        pass 