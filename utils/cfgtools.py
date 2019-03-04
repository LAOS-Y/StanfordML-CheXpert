import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# import sys
# sys.path.append("..")

import data
import model

import torchvision.transforms as transforms


def getTrm(trm_dict):
    if trm_dict["TYPE"] == "ColorJitter":
        return transforms.ColorJitter(brightness=trm_dict["BRIGHTNESS"],
                                      contrast=trm_dict["CONTRAST"],
                                      saturation=trm_dict["SATURATION"])
    elif trm_dict["TYPE"] == "RandomHorizontalFlip":
        return transforms.RandomHorizontalFlip(p=trm_dict["P"])
    elif trm_dict["TYPE"] == "RandomAffine":
        return transforms.RandomAffine(degrees=trm_dict["DEGREES"],
                                      translate=trm_dict["TRANSLATE"])
    else:
         assert False, "wrong PIL_TRANSFORMS type: {}".format(trm_dict["TYPE"])
            
def getDataset(ds_dict):
    if "PIL_TRANSFORMS" in ds_dict:
        trm_dict = ds_dict["PIL_TRANSFORMS"]
        
        trm = transforms.Compose([getTrm(trm) for trm in trm_dict])
    else:
        trm = None
    
    if ds_dict["NAME"] == "ChestXray-NIHCC":
        ds_train = data.ChestXray14(image_dir=ds_dict["IMAGE_DIR"], 
                                    csv_path=ds_dict["CSV_PATH"], 
                                    list_path=ds_dict["LIST_PATH_TRAIN"],
                                    downscale_shape=ds_dict["DOWNSCALE_SHAPE"],
                                    trm=trm,
                                    mean=ds_dict["NORM"]["MEAN"],
                                    std=ds_dict["NORM"]["STD"])
    
        ds_val = data.ChestXray14(image_dir=ds_dict["IMAGE_DIR"], 
                                    csv_path=ds_dict["CSV_PATH"], 
                                    list_path=ds_dict["LIST_PATH_VAL"],
                                    downscale_shape=ds_dict["DOWNSCALE_SHAPE"],
                                    trm=None,
                                    mean=ds_dict["NORM"]["MEAN"],
                                    std=ds_dict["NORM"]["STD"])
    
    elif ==[]
    
    else:
        assert False, "wrong DATASET name: {}".format(ds_dict["NAME"])
    
    return ds_train, ds_val

def getDataloader(ds_train, ds_val, dl_dict):
    bs_train = dl_dict["BATCH_PER_CARD_TRAIN"] * torch.cuda.device_count()
    bs_val = dl_dict["BATCH_PER_CARD_VAL"] * torch.cuda.device_count()
    
    dl_train = DataLoader(ds_train,
                          batch_size=bs_train,
                          shuffle=True,
                          num_workers=8)

    dl_val = DataLoader(ds_val, 
                        batch_size=bs_val, 
                        shuffle=True,
                        num_workers=8)
    
    return dl_train, dl_val

def getModel(model_dict):
    if model_dict["NAME"] == "Resnet50":
        net = model.Resnet50(model_dict["NUM_CLASS"]).cuda()
    elif model_dict["NAME"] == "Densenet121":
        net = model.Densenet121(model_dict["NUM_CLASS"]).cuda()
    else:
        assert False, "wrong MODEL name: {}".format(model_dict["NAME"])
    
    net = nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    return net

def getLoss(loss_dict):
    if loss_dict["NAME"] == "BCEWithLogitsLoss":
        BCE_WEIGHT = torch.Tensor(loss_dict["BCE_WEIGHT"])
        POS_WEIGHT = (1 - BCE_WEIGHT) / BCE_WEIGHT
        loss_fn = nn.BCEWithLogitsLoss(weight=BCE_WEIGHT, pos_weight=POS_WEIGHT).cuda()
    else:
        assert False, "wrong LOSS name: {}".format(loss_dict["NAME"])
    
    return loss_fn

def getOptimizer(net, optim_dict):
    LR = optim_dict["LR"]
    if optim_dict["NAME"] == "Adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=LR)
    else:
        assert False, "wrong OPTIMIZER name: {}".format(optim_dict["NAME"])
    
    return optimizer

def getScheduler(optimizer, sch_dict):
    if sch_dict["NAME"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=sch_dict["STEP_SIZE"],
                                                    gamma=sch_dict["DECAY_RATE"]) 
    else:
        assert False, "wrong SCHEDULER name: {}".format(sch_dict["NAME"])
    
    return scheduler