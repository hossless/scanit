import os
import cv2
import glob
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from utils import generate_enhancement_sample, generate_synthetic_sample

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

        return img_tensor, corners_tensor

class EnhancementDataset(Dataset):
    def __init__(self, dataset_dir):
        self.input_paths = sorted(glob.glob(os.path.join(dataset_dir, 'inputs', '*.jpg')))
        self.target_paths = sorted(glob.glob(os.path.join(dataset_dir, 'targets', '*.jpg')))

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_img = cv2.imread(self.input_paths[idx])
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB) 

        target_img = cv2.imread(self.target_paths[idx])
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)

        input_tensor = torch.from_numpy(input_img).float().permute(2, 0, 1) / 255.0
        target_tensor = torch.from_numpy(target_img).float().permute(2, 0, 1) / 255.0

        return input_tensor, target_tensor


class EndToEndDataset(Dataset):
    def __init__(self, dataset_dir):
        self.degraded_paths = sorted(glob.glob(os.path.join(dataset_dir, 'degraded', '*.jpg')))
        self.clean_paths = sorted(glob.glob(os.path.join(dataset_dir, 'clean', '*.jpg')))
        self.label_paths = sorted(glob.glob(os.path.join(dataset_dir, 'labels', '*.npy')))

    def __len__(self):
        return len(self.degraded_paths)

    def __getitem__(self, idx):
        degraded_512 = cv2.imread(self.degraded_paths[idx])
        degraded_512 = cv2.cvtColor(degraded_512, cv2.COLOR_BGR2RGB)
        
        degraded_256 = cv2.resize(degraded_512, (256, 256))
        
        clean_512 = cv2.imread(self.clean_paths[idx])
        clean_512 = cv2.cvtColor(clean_512, cv2.COLOR_BGR2RGB)
        
        corners = np.load(self.label_paths[idx])

        degraded_256_tensor = torch.from_numpy(degraded_256).float().permute(2, 0, 1) / 255.0
        degraded_512_tensor = torch.from_numpy(degraded_512).float().permute(2, 0, 1) / 255.0
        clean_512_tensor = torch.from_numpy(clean_512).float().permute(2, 0, 1) / 255.0
        corners_tensor = torch.from_numpy(corners).float()

        return degraded_256_tensor, degraded_512_tensor, corners_tensor, clean_512_tensor

def pre_generate_e2e_dataset(num_samples=8000, split="train"):
    base_dir = '/content/data'
    out_dir = f'/content/ready_dataset_e2e/{split}'
    os.makedirs(f'{out_dir}/degraded', exist_ok=True)
    os.makedirs(f'{out_dir}/clean', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)

    clean_scans = glob.glob(f'{base_dir}/scan/{split}/*.*')
    backgrounds = glob.glob(f'{base_dir}/background/*.*')

    print(f"🚀 Generating {num_samples} unified END-TO-END samples for {split}...")
    for i in tqdm(range(num_samples)):
        scan_path = random.choice(clean_scans)
        bg_path = random.choice(backgrounds)

        clean_scan = cv2.imread(scan_path)
        background_img = cv2.resize(cv2.imread(bg_path), (800, 800))

        degraded_img, ordered_corners, _ = generate_synthetic_sample(clean_scan, background_img)

        h, w = degraded_img.shape[:2]
        norm_corners = ordered_corners.copy()
        norm_corners[:, 0] /= w
        norm_corners[:, 1] /= h

        degraded_512 = cv2.resize(degraded_img, (512, 512))
        clean_512 = cv2.resize(clean_scan, (512, 512))

        cv2.imwrite(f'{out_dir}/degraded/sample_{i:05d}.jpg', degraded_512)
        cv2.imwrite(f'{out_dir}/clean/sample_{i:05d}.jpg', clean_512)
        np.save(f'{out_dir}/labels/sample_{i:05d}.npy', norm_corners)

def pre_generate_dataset(num_samples=10000, split="train"):
    base_dir = '/content/data'
    out_dir = f'/content/ready_dataset/{split}'
    os.makedirs(f'{out_dir}/images', exist_ok=True)
    os.makedirs(f'{out_dir}/labels', exist_ok=True)

    clean_scans = glob.glob(f'{base_dir}/scan/{split}/*.*')
    backgrounds = glob.glob(f'{base_dir}/background/*.*')

    print(f"Generating {num_samples} CORNER samples for {split}...")
    for i in tqdm(range(num_samples)):
        scan_path = random.choice(clean_scans)
        bg_path = random.choice(backgrounds)

        clean_scan = cv2.imread(scan_path)
        background_img = cv2.resize(cv2.imread(bg_path), (800, 800))

        degraded_img, corners, _ = generate_synthetic_sample(clean_scan, background_img)

        degraded_resized = cv2.resize(degraded_img, (256, 256))

        h, w = degraded_img.shape[:2]
        corners[:, 0] /= w
        corners[:, 1] /= h

        cv2.imwrite(f'{out_dir}/images/sample_{i:05d}.jpg', degraded_resized)
        np.save(f'{out_dir}/labels/sample_{i:05d}.npy', corners)

def pre_generate_enhancement_dataset(num_samples=10000, split="train"):
    base_dir = '/content/data'
    out_dir = f'/content/ready_dataset_enhancement/{split}'
    os.makedirs(f'{out_dir}/inputs', exist_ok=True)
    os.makedirs(f'{out_dir}/targets', exist_ok=True)

    clean_scans = glob.glob(f'{base_dir}/scan/{split}/*.*')
    backgrounds = glob.glob(f'{base_dir}/background/*.*')

    print(f"Generating {num_samples} ENHANCEMENT samples for {split}...")
    for i in tqdm(range(num_samples)):
        scan_path = random.choice(clean_scans)
        bg_path = random.choice(backgrounds)

        clean_scan = cv2.imread(scan_path)
        background_img = cv2.resize(cv2.imread(bg_path), (800, 800))

        degraded_input, clean_target = generate_enhancement_sample(clean_scan, background_img)

        cv2.imwrite(f'{out_dir}/inputs/sample_{i:05d}.jpg', degraded_input)
        cv2.imwrite(f'{out_dir}/targets/sample_{i:05d}.jpg', clean_target)

if __name__ == "__main__":
    pre_generate_e2e_dataset(num_samples=1000, split="train")
    pre_generate_e2e_dataset(num_samples=100, split="val")