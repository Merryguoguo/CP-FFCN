from cmath import nan
import pdb
import random
import torch
from PIL import Image
from glob import glob



class Places2(torch.utils.data.Dataset):
    def __init__(self, gt_root, cld_root, img_transform, split='Train'):
        super(Places2, self).__init__()
        self.img_transform = img_transform
    
        if split == 'Train':
            # -----------------------------------RICE2 (real cloudy)-----------------------------------
            # gt_img
            self.gt_paths = glob('{:s}/{:s}/*.png'.format(gt_root, split), 
                    recursive=True)          
            # simulated cloudy_img
            self.cloudy_paths = glob('{:s}/{:s}/*.png'.format(cld_root, split), 
                    recursive=True) 
        
        else:
            # -----------------------------------RICE2 (real cloudy)-----------------------------------
            self.gt_paths = glob('{:s}/{:s}/*.png'.format(gt_root, split)) 
            self.cloudy_paths = glob('{:s}/{:s}/*.png'.format(cld_root, split)) 
        
        if split == 'Real-Test':
            # -----------------------------------RICE2 (real cloudy)-----------------------------------
            self.paths = glob('{:s}/*.jpg'.format(gt_root), recursive=True)
            self.len = len(self.paths)
            self.train = int(len(self.paths) * 0.8)
            self.val = int(len(self.paths) * 0.1)
            self.test = int(len(self.paths) * 0.1)

            # gt_img
            self.gt_paths = glob('{:s}/*.jpg'.format(gt_root), 
                    recursive=True)[self.train+self.val: self.len]          
            # simulated cloudy_img
            self.cloudy_paths = glob('{:s}/*.jpg'.format(cld_root), 
                    recursive=True)[self.train+self.val: self.len]   

        if split == 'Real-Val':
            # -----------------------------------RICE2 (real cloudy)-----------------------------------
            self.paths = glob('{:s}/*.jpg'.format(gt_root), recursive=True)
            self.len = len(self.paths)
            self.train = int(len(self.paths) * 0.8)
            self.val = int(len(self.paths) * 0.1)
            self.test = int(len(self.paths) * 0.1)

            # gt_img
            self.gt_paths = glob('{:s}/*.jpg'.format(gt_root), 
                    recursive=True)[self.train : self.train+self.val]          
            # simulated cloudy_img
            self.cloudy_paths = glob('{:s}/*.jpg'.format(cld_root), 
                    recursive=True)[self.train : self.train+self.val]  

        if split == 'Real-Train':
            # -----------------------------------RICE2 (real cloudy)-----------------------------------
            self.paths = glob('{:s}/*.jpg'.format(gt_root), recursive=True)
            self.len = len(self.paths)
            self.train = int(len(self.paths) * 0.8)
            self.val = int(len(self.paths) * 0.1)
            self.test = int(len(self.paths) * 0.1)

            # gt_img
            self.gt_paths = glob('{:s}/*.jpg'.format(gt_root), 
                    recursive=True)[0 : self.train]          
            # simulated cloudy_img
            self.cloudy_paths = glob('{:s}/*.jpg'.format(cld_root), 
                    recursive=True)[0 : self.train]  

    def __getitem__(self, index):
        gt_img = Image.open(self.gt_paths[index])
        gt_img = self.img_transform(gt_img.convert('RGB'))

        cld_img = Image.open(self.cloudy_paths[index])
        cld_img = self.img_transform(cld_img.convert('RGB'))
        # print('index:', index)
        # print(self.gt_paths[index])
        # print(self.cloudy_paths[index])

        return cld_img, gt_img


    def __len__(self):
        return len(self.gt_paths)


    def __paths__(self):
        return self.cloudy_paths, self.gt_paths
