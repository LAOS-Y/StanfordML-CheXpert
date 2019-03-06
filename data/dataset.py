import pandas as pd
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
from PIL import Image

class ChestXray14(Dataset):
    def __init__(self, image_dir, csv_path, list_path, trm=None, downscale_shape=(224, 224), mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super(ChestXray14, self).__init__()
        self.image_dir = image_dir
        self.csv_path = csv_path
        self.list_path = list_path
        
        trm_norm = transforms.Normalize(mean=mean, std=std)
        trm_resize = transforms.Resize(downscale_shape)
        
        if trm == None:
            self.trm = transforms.ToTensor()
        else:
            self.trm = transforms.Compose([trm, transforms.ToTensor()])
            
        self.trm = transforms.Compose([trm_resize, self.trm, trm_norm])
        
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia','Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', "Hernia"]
        
        self.img_list = []
        self.label_list = []
        
        csv_df = pd.read_csv(csv_path)
        raw_label = list(csv_df['Finding Labels'])
        img_idx = list(csv_df['Image Index'])
        
        with open(list_path) as file:
            self.filenames = file.readlines()
            for i, name in enumerate(self.filenames):
                self.filenames[i] = name.strip('\n')
        
        for name in self.filenames:
            i = img_idx.index(name)

            label_list = raw_label[i].split('|')
            np_label = np.zeros(14)
            
            for label in label_list:
                if label == 'No Finding':
                    break
                else:
                    np_label[self.classes.index(label)] = 1

            self.label_list.append(torch.Tensor(np_label))
        
        assert len(self.filenames) == len(self.label_list) 
        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        name = self.filenames[index]
        img = Image.open(self.image_dir + name).convert("RGB")
        return self.trm(img), self.label_list[index]#, self.filenames[index]
