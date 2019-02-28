import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from model import Densenet121
import data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

import utils

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torchvision.transforms as transforms

trm_color = transforms.ColorJitter(brightness=0.01,
                                   contrast=0.01,
                                   saturation=0.01)

trm_h_flip = transforms.RandomHorizontalFlip(p=0.3)
trm_affine = transforms.RandomAffine(3, (0.05, 0.05))

trm = transforms.Compose([trm_h_flip, trm_affine, trm_color, transforms.ToTensor()])

BATCH_SIZE_PER_CARD = 8
BATCH_SIZE = BATCH_SIZE_PER_CARD * torch.cuda.device_count()

BCE_WEIGHT = torch.Tensor([0.0957, 0.0197, 0.1001, 0.1593, 0.0466, 0.0544, 0.0101, 0.0305, 0.0330,
                           0.0159, 0.0164, 0.0145, 0.0259, 0.0016])

POS_WEIGHT = (1 - BCE_WEIGHT) / BCE_WEIGHT

ds_train = data.ChestXray14("dataset/ChestXray-NIHCC/images/", 
                            "dataset/ChestXray-NIHCC/Data_Entry_2017.csv", 
                            "dataset/ChestXray-NIHCC/train_val_list.txt",
                            trm=trm)

ds_val = data.ChestXray14("dataset/ChestXray-NIHCC/images/", 
                          "dataset/ChestXray-NIHCC/Data_Entry_2017.csv", 
                          "dataset/ChestXray-NIHCC/test_list.txt")

dl_train = DataLoader(ds_train, 
                      batch_size=BATCH_SIZE, 
                      shuffle=True,
                      num_workers=8)

dl_val = DataLoader(ds_val, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True,
                    num_workers=8)

loss_fn = nn.BCEWithLogitsLoss(weight=BCE_WEIGHT, pos_weight=POS_WEIGHT).cuda()

net = Densenet121(14).cuda()
net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))

MAX_NO_OPTIM = 1
DECAY_RATE = 0.1

optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                            step_size=MAX_NO_OPTIM,
                                            gamma=DECAY_RATE) 

NUM_EPOCH = 30

loss_train_list = []
loss_val_list = []
optimizer.zero_grad()

LOG_PATH = "train_log/log6/"

solver = utils.Solver(net, dl_train, dl_val, loss_fn, optimizer, scheduler, LOG_PATH)

for i in range(NUM_EPOCH):
    with open(LOG_PATH + "log.txt", mode="a") as file:
        print("Epoch#{}:".format(i + 1))
        print("Epoch#{}:".format(i + 1), file=file)
        
    solver.optimize()
    solver.validate()
    
    solver.save_model("Epoch#{}.th".format(i + 1))
#     if (len(self.loss_val_list) > 1) and (self.loss_val_list[-2] < self.loss_val_list[-1]):
    solver.update_lr()
    
solver.save_loss_csv("loss.csv")
solver.plot_loss("loss.jpg")