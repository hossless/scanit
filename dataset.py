import os
import cv2
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import generate_synthetic_sample

class DocumentDataset(Dataset):
    def __init__(self, dataset_dir):
        self.image_paths = sorted(glob.glob(os.path.join(dataset_dir, 'images', '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(dataset_dir, 'labels', '*.npy')))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        
        corners = np.load(self.label_paths[idx])
        
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        corners_tensor = torch.from_numpy(corners).float().flatten()
        
        dummy_clean = torch.zeros_like(img_tensor) 
        
        return img_tensor, dummy_clean, corners_tensor
    
def pre_generate_dataset(num_samples=10000, split="train"):
    base_dir = '/content/data'
    out_dir = f'/content/ready_dataset/{split}'
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)
    
    clean_scans = glob.glob(f'{base_dir}/scan/{split}/*.*')
    backgrounds = glob.glob(f'{base_dir}/background/*.*')
    
    for i in tqdm(range(num_samples)):
        scan_path = random.choice(clean_scans)
        bg_path = random.choice(backgrounds)
        
        clean_scan = cv2.imread(scan_path)
        background_img = cv2.resize(cv2.imread(bg_path), (800, 800))
        
        degraded_img, corners = generate_synthetic_sample(clean_scan, background_img)
        
        degraded_resized = cv2.resize(degraded_img, (256, 256))
        
        h, w = degraded_img.shape[:2]
        corners[:, 0] /= w
        corners[:, 1] /= h
        
        cv2.imwrite(f'{out_dir}/images/sample_{i:05d}.jpg', degraded_resized)
        np.save(f'{out_dir}/labels/sample_{i:05d}.npy', corners)

if __name__ == "__main__":
    pre_generate_dataset(num_samples=8000, split="train")
    pre_generate_dataset(num_samples=200, split="val")