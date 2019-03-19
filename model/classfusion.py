import torch
import torch.nn as nn

class ClassFusion(nn.Module):
    def __init__(self, num_class=14, in_c=14, h_c=6, initer=None):
        super(ClassFusion, self).__init__()
        self.fc1 = nn.Linear(in_features=in_c, out_features=h_c)
        self.fc2 = nn.Linear(in_features=h_c, out_features=num_class)
    
        self.initer = initer
        if self.initer is not None:
            self.initer(self.fc1.weight)
            self.initer(self.fc2.weight)

    def forward(self, x):
        out = self.fc1(x)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        out = nn.Sigmoid()(out)
        
        out = x * out
        return out
