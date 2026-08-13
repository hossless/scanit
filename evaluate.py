import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from model import UNetEnhancer
from pytorch_msssim import ssim 
from dataset import DocumentDataset
from dataset import EnhancementDataset
from torch.utils.data import DataLoader
from model import DirectRegressionNet, HeatmapCornerNet
from engine import order_points, extract_coords_from_heatmap

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

def calculate_corner_errors(pred_coords, target_coords):
    pred_coords = np.array(pred_coords)    
    pred_coords = pred_coords.reshape(4, 2)
    target_coords = target_coords.reshape(4, 2)
    
    
    pred_ordered = order_points(pred_coords)
    target_ordered = order_points(target_coords)
    
    
    distances = np.linalg.norm(pred_ordered - target_ordered, axis=1)
    
    mean_error = np.mean(distances)
    max_error = np.max(distances) 
    
    return mean_error, max_error

def evaluate_corner_model(model, model_type, dataloader, device, threshold=10.0):
    total_mean_error = 0.0
    successful_images = 0
    total_images = 0
    
    with torch.no_grad():
        for images, targets in dataloader:
            images, targets = images.to(device), targets.to(device)
            predictions = model(images)
            
            
            for i in range(images.size(0)):
                if model_type == "direct":
                    
                    pred_coords = (predictions[i].cpu().numpy() * 256.0)
                else:
                    
                    pred_coords = extract_coords_from_heatmap(predictions[i].cpu().numpy())
                    
                
                target_coords = (targets[i].cpu().numpy() * 256.0)
                
                mean_err, max_err = calculate_corner_errors(pred_coords, target_coords)
                
                total_mean_error += mean_err
                if max_err <= threshold:
                    successful_images += 1
                total_images += 1
                
    avg_error = total_mean_error / total_images
    success_rate = (successful_images / total_images) * 100.0
    
    return avg_error, success_rate

def generate_corner_table(direct_version="direct_v2", heatmap_version="heatmap_v2"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🚀 Starting Corner Model Showdown on {device}...")
    
    
    batch_size = 16
    test_loader = DataLoader(DocumentDataset('/content/ready_dataset/test'), batch_size=batch_size, shuffle=False)
    
    
    direct_model = DirectRegressionNet().to(device)
    direct_path = os.path.join("models", direct_version, "direct_best.pth")
    if os.path.exists(direct_path):
        direct_model.load_state_dict(torch.load(direct_path, map_location=device, weights_only=True))
        direct_model.eval()
        direct_err, direct_succ = evaluate_corner_model(direct_model, "direct", test_loader, device)
    else:
        print(f"❌ Could not find {direct_version}! Skipping.")
        direct_err, direct_succ = 0.0, 0.0

    
    heatmap_model = HeatmapCornerNet().to(device)
    heatmap_path = os.path.join("models", heatmap_version, "heatmap_best.pth")
    if os.path.exists(heatmap_path):
        heatmap_model.load_state_dict(torch.load(heatmap_path, map_location=device, weights_only=True))
        heatmap_model.eval()
        heatmap_err, heatmap_succ = evaluate_corner_model(heatmap_model, "heatmap", test_loader, device)
    else:
        print(f"❌ Could not find {heatmap_version}! Skipping.")
        heatmap_err, heatmap_succ = 0.0, 0.0

    
    print("\n" + "="*60)
    print("🎯 CORNER DETECTION PERFORMANCE (Synthetic Test Set)")
    print("="*60)
    print(f"| {'Model Type':<20} | {'Mean Pixel Error':<18} | {'Success Rate (<10px)':<20} |")
    print("-" * 64)
    print(f"| {'Direct Regression':<20} | {direct_err:>13.2f} px    | {direct_succ:>18.1f} % |")
    print(f"| {'Heatmap Regression':<20} | {heatmap_err:>13.2f} px    | {heatmap_succ:>18.1f} % |")
    print("="*60 + "\n")

if __name__ == "__main__":
    # generate_enhancer_table(enhancer_version="enhancer_v2")
    generate_corner_table(direct_version="direct_v2", heatmap_version="heatmap_v2")