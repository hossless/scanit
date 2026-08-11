import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


from dataset import DocumentDataset, EnhancementDataset
from model import DirectRegressionNet, HeatmapCornerNet, UNetEnhancer
from utils import generate_target_heatmaps

def dry_run_corner_models(model_type="direct"):
    batch_size = 4
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Starting {model_type.upper()} dry run on {device} ---")

    
    dataset = DocumentDataset('./ready_dataset/train') 
    
    
    if len(dataset) == 0:
        print("⚠️ No data found in ./ready_dataset/train! Generate a few samples locally first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    if model_type == "direct":
        model = DirectRegressionNet().to(device)
        criterion = nn.L1Loss() 
    elif model_type == "heatmap":
        model = HeatmapCornerNet().to(device)
        criterion = nn.MSELoss()
    else:
        raise ValueError("model_type must be 'direct' or 'heatmap'")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    
    for batch_idx, (degraded_imgs, corners) in enumerate(dataloader):
        degraded_imgs = degraded_imgs.to(device)
        corners = corners.to(device) 

        if model_type == "heatmap":
            targets = generate_target_heatmaps(corners, image_size=256)
        else:
            targets = corners

        predictions = model(degraded_imgs)
        loss = criterion(predictions, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
        break 

    print(f"✅ Dry run complete! {model_type.upper()} architecture is perfectly wired!")

def dry_run_enhancer():
    
    batch_size = 2 
    learning_rate = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Starting ENHANCER dry run on {device} ---")

    dataset = EnhancementDataset('./ready_dataset_enhancement/train')
    
    if len(dataset) == 0:
        print("⚠️ No data found in ./ready_dataset_enhancement/train! Generate a few samples locally first.")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = UNetEnhancer().to(device)
    
    criterion = nn.MSELoss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    
    for batch_idx, (input_imgs, target_imgs) in enumerate(dataloader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        predictions = model(input_imgs)
        loss = criterion(predictions, target_imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")
        break 

    print(f"✅ Dry run complete! ENHANCER architecture is perfectly wired!")

if __name__ == "__main__":
    
    dry_run_corner_models(model_type="direct")
    dry_run_corner_models(model_type="heatmap")
    dry_run_enhancer()