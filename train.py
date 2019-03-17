import numpy as np
import os

import torch
import torch.nn as nn

import utils

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("config_path", help="path to the yml config file")
parser.add_argument("-v", "--verbose", help="verbosely print the process", action="store_true")
args = parser.parse_args()

yml_dict = yaml.load(open(args.config_path))

os.environ["CUDA_VISIBLE_DEVICES"] = yml_dict["CUDA_VISIBLE_DEVICES"]

LOG_DIR = yml_dict["LOG_DIR"]

log_file = open(LOG_DIR + "log.txt", mode="a") 

log_writer = utils.LogWriter(log_file)

log_writer.write(text="Start initializing dataset",
                 verbose=args.verbose)

ds_train, ds_val = utils.getDataset(ds_dict=yml_dict["DATASET"][0])

log_writer.write(text="Finish initializing dataset",
                 verbose=args.verbose)

dl_train, dl_val = utils.getDataloader(ds_train=ds_train,
                                       ds_val=ds_val,
                                       dl_dict=yml_dict["DATALOADER"])

net = utils.getModel(yml_dict["MODEL"])
loss_fn = utils.getLoss(yml_dict["LOSS"])
optimizer = utils.getOptimizer(net, yml_dict["OPTIMIZER"])
scheduler = utils.getScheduler(optimizer, yml_dict["SCHEDULER"])

NUM_EPOCH = yml_dict["NUM_EPOCH"]
CSV_SAVING_MARGIN = max(NUM_EPOCH // 20, 1)

solver = utils.Solver(net, dl_train, dl_val, loss_fn, optimizer, scheduler, LOG_DIR, log_file, ds_val.classes)

log_writer.write(text="Start training",
                 verbose=args.verbose)

for i in range(NUM_EPOCH): 
    log_writer.write(text="Epoch#{}/{}:".format(i + 1, NUM_EPOCH),
                     verbose=args.verbose)

    solver.optimize(verbose=args.verbose)
    solver.validate(verbose=args.verbose)
    
    if solver.loss_val_list[-1] <= min(solver.loss_val_list):
        solver.save_model("Epoch#{}.th".format(i + 1), verbose=args.verbose)
        solver.plot_roc("roc.jpg", model_name=yml_dict["MODEL"]["NAME"], verbose=args.verbose)
    
    solver.update_lr(metric=solver.loss_val_list[-1],
                     verbose=args.verbose)

    if i % CSV_SAVING_MARGIN == 0: 
        solver.save_loss_csv("loss.csv", verbose=args.verbose)
        solver.plot_loss("loss.jpg", verbose=args.verbose)

log_writer.write(text="Finish training",
                 verbose=args.verbose)
        
solver.save_loss_csv("loss.csv", verbose=args.verbose)
solver.plot_loss("loss.jpg", verbose=args.verbose)
