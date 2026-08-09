import os
import cv2
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from utils import generate_synthetic_sample

class DocumentDataset(Dataset):
    def __init__(self, clean_scans_dir, backgrounds_dir, image_size=256, epoch_size=1000):
        self.clean_scans = glob.glob(os.path.join(clean_scans_dir, '*.*'))
        self.backgrounds = glob.glob(os.path.join(backgrounds_dir, '*.*'))
        
        self.image_size = image_size
        
        self.epoch_size = epoch_size 

    def __len__(self):
        return self.epoch_size

    def __getitem__(self, idx):
        scan_path = random.choice(self.clean_scans)
        bg_path = random.choice(self.backgrounds)
        
        clean_scan = cv2.imread(scan_path)
        background_img = cv2.imread(bg_path)
        
        background_img = cv2.resize(background_img, (800, 800))
        
        degraded_img, corners = generate_synthetic_sample(clean_scan, background_img)
        
        orig_h, orig_w = degraded_img.shape[:2]
        
        degraded_resized = cv2.resize(degraded_img, (self.image_size, self.image_size))
        clean_resized = cv2.resize(clean_scan, (self.image_size, self.image_size))
        
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h
        corners[:, 0] *= scale_x
        corners[:, 1] *= scale_y
        
        corners_normalized = corners / self.image_size
        
        degraded_tensor = torch.from_numpy(degraded_resized).float() / 255.0
        clean_tensor = torch.from_numpy(clean_resized).float() / 255.0
        
        degraded_tensor = degraded_tensor.permute(2, 0, 1)
        clean_tensor = clean_tensor.permute(2, 0, 1)
        
        corners_tensor = torch.from_numpy(corners_normalized).float().flatten()
        
        return degraded_tensor, clean_tensor, corners_tensor