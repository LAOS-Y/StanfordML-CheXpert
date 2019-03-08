import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from .logwriter import LogWriter

class Solver():
    def __init__(self, net, dl_train, dl_val, loss_fn, optimizer, scheduler, log_path, log_file):
        self.loss_train_list = []
        self.loss_val_list = []
        self.lr_list = []
        
        self.dl_train = dl_train
        self.dl_val = dl_val
        
        self.net = net
        
        self.log_path = log_path
        self.log_writer = LogWriter(log_file)
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def optimize(self, verbose=False):
        cur_loss_train = 0
        ds_len = 0
        
        self.net.train()
        
        for img, label in self.dl_train:
            img = img.cuda()
            label = label.cuda()
            pred = self.net(img)
            loss = self.loss_fn(pred, label)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_np = loss.detach().cpu().numpy()
            cur_loss_train = cur_loss_train + loss_np * img.shape[0]
            ds_len += img.shape[0]

        cur_loss_train /= ds_len
        self.loss_train_list.append(cur_loss_train)
        self.lr_list.append(self.optimizer.param_groups[0]['lr'])

        self.log_writer.write(text="Train loss: {}".format(cur_loss_train),
                              verbose=verbose)
        
    def validate(self, verbose=False):
        cur_loss_val = 0
        ds_len = 0
        
        self.net.eval()

        for img, label in self.dl_val:
            img = img.cuda()
            label = label.cuda()
            pred = self.net(img)
            loss = self.loss_fn(pred, label)

            loss_np = loss.detach().cpu().numpy()
            cur_loss_val += loss_np * img.shape[0]
            ds_len += img.shape[0]

        cur_loss_val /= ds_len
        self.loss_val_list.append(cur_loss_val)

        self.log_writer.write(text="Val loss: {}".format(cur_loss_val),
                              verbose=verbose)
        
    def save_model(self, filename, verbose=False):
        torch.save(self.net.state_dict(), self.log_path + filename)
        
        self.log_writer.write(text="Weights saved as {}".format(self.log_path + filename),
                              verbose=verbose)
        
    def update_lr(self, metric, verbose=False):
        pre_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step(metric)
        cur_lr = self.optimizer.param_groups[0]['lr']
        
        if cur_lr != pre_lr:
            self.log_writer.write(text="Learning rate decayed to {}".format(cur_lr),
                                  verbose=verbose)
        
    def plot_loss(self, filename, dpi=200, show=False, verbose=False):
        plt.rcParams["figure.dpi"] = dpi

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        
        plt.plot(self.loss_train_list, label="loss_train", color="r")
        plt.plot(self.loss_val_list, label="loss_val", color="g")
        
        plt.legend(fontsize='x-small', loc='best')
        
        plt.twinx()
        plt.ylabel("Learning Rate")
        plt.plot(self.lr_list, label="lr", color="b")

        plt.legend(fontsize='x-small', loc='center right')
        
        plt.savefig(self.log_path + "loss.jpg")
        if show:
            plt.show()
            
        plt.close("all")

        self.log_writer.write(text="Loss Plotted at {}".format(self.log_path + filename),
                              verbose=verbose)
            
    def save_loss_csv(self, filename, verbose=False):
        df = pd.DataFrame({"Epoch": np.arange(len(self.loss_train_list)) + 1,
                           "Loss_train": self.loss_train_list,
                           "Loss_val": self.loss_val_list,
                           "lr": self.lr_list})
        
        df.to_csv(self.log_path + filename, index=False)
        
        self.log_writer.write(text="Loss saved at {}".format(self.log_path + filename),
                              verbose=verbose)
