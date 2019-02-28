import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class Solver():
    def __init__(self, net, dl_train, dl_val, loss_fn, optimizer, scheduler, log_path):
        self.loss_train_list = []
        self.loss_val_list = []
        self.lr_list = []
        
        self.dl_train = dl_train
        self.dl_val = dl_val
        
        self.net = net
        
        self.log_path = log_path
        
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        
    def optimize(self, verbose=True):
        cur_loss_train = 0
        ds_len = 0
        
        self.net.train()
        
        with open(self.log_path+ "log.txt", mode="a") as file:
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
            
            print("Train loss: {}".format(cur_loss_train), file=file)
            if verbose:
                print("Train loss: {}".format(cur_loss_train))

    def validate(self, verbose=True):
        cur_loss_val = 0
        ds_len = 0
        
        self.net.eval()
        with open(self.log_path+ "log.txt", mode="a") as file:
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

            print("Val loss: {}".format(cur_loss_val), file=file)
            if verbose:
                print("Val loss: {}".format(cur_loss_val))
        
    def save_model(self, filename, verbose=True):
        with open(self.log_path+ "log.txt", mode="a") as file:
            torch.save(self.net.state_dict(), self.log_path + filename)
            print("Weights saved as {}".format(self.log_path + filename), file=file)
            print("Weights saved as {}".format(self.log_path + filename))
        
        
    def update_lr(self, verbose=True):
        with open(self.log_path+ "log.txt", mode="a") as file:
            self.scheduler.step()
            
            cur_lr = self.optimizer.param_groups[0]['lr']
            
            if (len(self.lr_list) == 0) or (cur_lr != self.lr_list[-1]):
                print("Learning rate decayed to {}".format(cur_lr), file=file)
                if verbose:
                    print("Learning rate decayed to {}".format(cur_lr))
        
    def plot_loss(self, filename, dpi=200, show=False, verbose=True):
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

        with open(self.log_path+ "log.txt", mode="a") as file:
            print("Loss Plotted at {}".format(self.log_path + filename), file=file)
            if verbose:
                print("Loss Plotted at {}".format(self.log_path + filename))
            
    def save_loss_csv(self, filename, verbose=True):
        df = pd.DataFrame({"Epoch": np.arange(len(self.loss_train_list)) + 1,
                           "Loss_train": self.loss_train_list,
                           "Loss_val": self.loss_val_list,
                           "lr": self.lr_list})
        
        df.to_csv(self.log_path + filename, index=False)
        
        with open(self.log_path+ "log.txt", mode="a") as file:
            print("Loss saved at {}".format(self.log_path + filename), file=file)
            if verbose:
                print("Loss saved at {}".format(self.log_path + filename))