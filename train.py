import numpy as np
import os

import torch
import torch.nn as nn

import utils
import data

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="path to the yml config file")
parser.add_argument("-v", "--verbose", help="verbosely print the process", action="store_true")
args = parser.parse_args()

yml_dict = yaml.load(open(args.config_path))

os.environ["CUDA_VISIBLE_DEVICES"] = yml_dict["CUDA_VISIBLE_DEVICES"]

LOG_DIR = yml_dict["LOG_DIR"]

with open(LOG_DIR + "log.txt", mode="a") as file:
    print("Start initializing dataset", file=file)
if args.verbose:
    print("Start initializing dataset")

ds_train, ds_val = utils.getDataset(ds_dict=yml_dict["DATASET"][0])

with open(LOG_DIR + "log.txt", mode="a") as file:
    print("Finish initializing dataset", file=file)  
if args.verbose:
    print("Finish initializing dataset")

dl_train, dl_val = utils.getDataloader(ds_train=ds_train,
                                 ds_val=ds_val,
                                 dl_dict=yml_dict["DATALOADER"])

loss_fn = utils.getLoss(yml_dict["LOSS"])

net = utils.getModel(yml_dict["MODEL"])

optimizer = utils.getOptimizer(net, yml_dict["OPTIMIZER"])
scheduler = utils.getScheduler(optimizer, yml_dict["SCHEDULER"])

NUM_EPOCH = yml_dict["NUM_EPOCH"]

solver = utils.Solver(net, dl_train, dl_val, loss_fn, optimizer, scheduler, LOG_DIR)

with open(LOG_DIR + "log.txt", mode="a") as file:
    print("Start training", file=file)
if args.verbose:
    print("Start training")

for i in range(NUM_EPOCH):
    with open(LOG_DIR + "log.txt", mode="a") as file: 
        print("Epoch#{}/{}:".format(i + 1, NUM_EPOCH), file=file)
    if args.verbose:
        print("Epoch#{}/{}:".format(i + 1, NUM_EPOCH))
        
    solver.optimize(verbose=args.verbose)
    solver.validate(verbose=args.verbose)
    
    solver.save_model("Epoch#{}.th".format(i + 1), verbose=args.verbose)
    
    if (len(solver.loss_val_list) > 1) and (solver.loss_val_list[-2] < solver.loss_val_list[-1]):
        solver.update_lr(verbose=args.verbose)

with open(LOG_DIR + "log.txt", mode="a") as file:
    print("Finish training", file=file)
if args.verbose:
    print("Finish training")
    
solver.save_loss_csv("loss.csv", verbose=args.verbose)
solver.plot_loss("loss.jpg", verbose=args.verbose)
