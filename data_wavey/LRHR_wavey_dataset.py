# %%
import  os.path
import  numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
import io, requests

from datetime import datetime



class LRHR_WAVEY_Dataset(Dataset):

    def __init__(self, dataroot, split,scale_factor):
        
        data = np.load(dataroot)
        if split == 'train':
            Hy_fields = data['Hy_fields'].astype(np.float32)[0:24000,...]
        elif split == 'val':
            Hy_fields = data['Hy_fields'].astype(np.float32)[24000:,...]
        self.data_len = Hy_fields.shape[0]
        self.imgs_HR = torch.tensor(Hy_fields[:,:,:,0:64])
        min_max = [self.imgs_HR.min(), self.imgs_HR.max()]
        self.imgs_LR = nn.functional.interpolate(self.imgs_HR,scale_factor=scale_factor,mode='bicubic')
        self.imgs_SR = nn.functional.interpolate(self.imgs_LR,scale_factor=1/scale_factor,mode='bicubic')
        self.imgs_HR = 2*((self.imgs_HR-min_max[0])/(min_max[1]-min_max[0])) - 1
        self.imgs_LR = 2*((self.imgs_LR-min_max[0])/(min_max[1]-min_max[0])) - 1
        self.imgs_SR = 2*((self.imgs_SR-min_max[0])/(min_max[1]-min_max[0])) - 1

    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):

        img_HR = self.imgs_HR[index]
        img_LR = self.imgs_LR[index]
        img_SR = self.imgs_SR[index]

        return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index} 
    


# dataset = LRHR_WAVEY_Dataset('/home/yuxinlin/Deep_EM/WaveYNet/train_ds.npz',0.25) 
# # %%
# print(dataset.__getitem__(0)['LR'].shape)
# print(dataset.__getitem__(0)['SR'].shape)
# print(dataset.__getitem__(0)['HR'].shape)
# print(dataset.__getitem__(0)['LR'].min())
# print(dataset.__getitem__(0)['SR'].min())
# print(dataset.__getitem__(0)['HR'].min())
# print(dataset.__getitem__(0)['LR'].max())
# print(dataset.__getitem__(0)['SR'].max())
# print(dataset.__getitem__(0)['HR'].max())
# # %%

# %%
