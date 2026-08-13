import os
import torch
import numpy as np
import torch.nn.functional as F
from model import UNetEnhancer
from pytorch_msssim import ssim 
from dataset import EnhancementDataset
from torch.utils.data import DataLoader

def calculate_psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def evaluate_split(dataloader, model, device):
    total_baseline_psnr = 0.0
    total_baseline_ssim = 0.0
    total_model_psnr = 0.0
    total_model_ssim = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            
            
            baseline_psnr = calculate_psnr(inputs, targets)
            baseline_ssim = ssim(inputs, targets, data_range=1.0, size_average=True)
            
            total_baseline_psnr += baseline_psnr.item()
            total_baseline_ssim += baseline_ssim.item()
            
            
            
            if model is not None:
                preds = model(inputs)
                model_psnr = calculate_psnr(preds, targets)
                model_ssim = ssim(preds, targets, data_range=1.0, size_average=True)
                
                total_model_psnr += model_psnr.item()
                total_model_ssim += model_ssim.item()
                
            num_batches += 1
            
    avg_base_psnr = total_baseline_psnr / num_batches
    avg_base_ssim = total_baseline_ssim / num_batches
    avg_model_psnr = total_model_psnr / num_batches if model else 0.0
    avg_model_ssim = total_model_ssim / num_batches if model else 0.0
    
    return avg_base_psnr, avg_base_ssim, avg_model_psnr, avg_model_ssim

def generate_enhancer_table(enhancer_version="enhancer_v2"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting Enhancer Evaluation on {device}...")
    
    
    model = UNetEnhancer().to(device)
    model_path = os.path.join("models", enhancer_version, "enhancer_best.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()
        print(f"✅ Loaded {enhancer_version} weights!")
    else:
        print(f"❌ Could not find weights at {model_path}! Evaluating Baseline only.")
        model = None

    
    batch_size = 8
    base_dir = '/content/ready_dataset_enhancement'
    
    splits = {
        "Training": DataLoader(EnhancementDataset(f'{base_dir}/train'), batch_size=batch_size, shuffle=False),
        "Validation": DataLoader(EnhancementDataset(f'{base_dir}/val'), batch_size=batch_size, shuffle=False),
        "Test": DataLoader(EnhancementDataset(f'{base_dir}/test'), batch_size=batch_size, shuffle=False)
    }

    
    results = {}
    for split_name, loader in splits.items():
        print(f"🔍 Evaluating {split_name} split...")
        results[split_name] = evaluate_split(loader, model, device)

    
    print("\n" + "="*50)
    print("📊 ENHANCEMENT MODEL PERFORMANCE TABLE")
    print("="*50)
    print(f"| {'Split':<12} | {'Baseline PSNR':<13} | {'Model PSNR':<10} | {'Baseline SSIM':<13} | {'Model SSIM':<10} |")
    print("-" * 68)
    
    for split in ["Training", "Validation", "Test"]:
        b_psnr, b_ssim, m_psnr, m_ssim = results[split]
        print(f"| {split:<12} | {b_psnr:>13.2f} | {m_psnr:>10.2f} | {b_ssim:>13.4f} | {m_ssim:>10.4f} |")
        
    print("="*50 + "\n")
    

if __name__ == "__main__":
    generate_enhancer_table(enhancer_version="enhancer_v2")