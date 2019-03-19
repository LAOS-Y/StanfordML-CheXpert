import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# import sys
# sys.path.append("..")

from .data import *
import model
from collections import OrderedDict

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
        ds_train = ChestXray14(image_dir=ds_dict["IMAGE_DIR"], 
                               csv_path=ds_dict["CSV_PATH"], 
                               list_path=ds_dict["LIST_PATH_TRAIN"],
                               downscale_shape=ds_dict["DOWNSCALE_SHAPE"],
                               trm=trm,
                               mean=ds_dict["NORM"]["MEAN"],
                               std=ds_dict["NORM"]["STD"])
    
        ds_val = ChestXray14(image_dir=ds_dict["IMAGE_DIR"], 
                             csv_path=ds_dict["CSV_PATH"], 
                             list_path=ds_dict["LIST_PATH_VAL"],
                             downscale_shape=ds_dict["DOWNSCALE_SHAPE"],
                             trm=None,
                             mean=ds_dict["NORM"]["MEAN"],
                             std=ds_dict["NORM"]["STD"])
    
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

def getIniter(init_dict):
    if init_dict["NAME"] == "kaiming_normal_":
        a = init_dict["A"]
        mode = init_dict["MODE"]
        nonlinearity = init_dict["NONLINEARITY"]

        return lambda x: nn.init.kaiming_normal_(x,
                                                 a=a,
                                                 mode=mode,
                                                 nonlinearity=nonlinearity)
    elif init_dict["NAME"] == "xavier_normal_":
        gain = init_dict["GAIN"]
    
        return lambda x: nn.init.xavier_normal_(x,
                                                gain=gain)
    else:
        assert False, "wrong INITER name: {}".format(model_dict["NAME"])

def getSingleModel(model_dict):
    backbone_class_dict = {"Resnet50":model.Resnet50,
                           "Densenet121":model.Densenet121,
                           "SEResnet50":model.SEResnet50,
                           "SEDensenet121":model.SEDensenet121}

    name = model_dict["NAME"]
    
    if name in backbone_class_dict:
        num_class = model_dict["NUM_CLASS"]
        pretrained = model_dict["PRETRAINED"]
        initer = getIniter(model_dict["INITER"])

        net = backbone_class_dict[name](num_class=num_class,
                                        pretrained=pretrained,
                                        initer=initer)
    elif name == "ClassFusion":
        num_class = model_dict["NUM_CLASS"]
        in_c = model_dict["IN_C"]
        h_c = model_dict["H_C"]
        initer = getIniter(model_dict["INITER"])

        net = model.ClassFusion(num_class=num_class,
                          in_c=in_c,
                          h_c=h_c,
                          initer=initer)
    else:
        assert False, "wrong Model name: {}".format(name)

    if "WEIGHT_PATH" in model_dict:
        weight_path = model_dict["WEIGHT_PATH"]
        net.load_state_dict(torch.load(weight_path))

    if model_dict["FREEZE"]:
        for param in net.parameters():
            param.requires_grad = False
    return net.cuda()

def getModel(model_dicts):
    models = OrderedDict([(model_dict["NAME"], getSingleModel(model_dict)) for model_dict in model_dicts])

    net = nn.Sequential(models)

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
    elif sch_dict["NAME"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode=sch_dict["MODE"],
                                                               factor=sch_dict["FACTOR"],
                                                               patience=sch_dict["PATIENCE"],
                                                               verbose=sch_dict["VERBOSE"],
                                                               threshold=sch_dict["THRESHOLD"],
                                                               threshold_mode=sch_dict["THRESHOLD_MODE"],
                                                               cooldown=sch_dict["COOLDOWN"],
                                                               min_lr=sch_dict["MIN_LR"],
                                                               eps=sch_dict["EPS"])
    
    else:
        assert False, "wrong SCHEDULER name: {}".format(sch_dict["NAME"])
    
    return scheduler
