import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import DocumentDataset
from torch.utils.data import DataLoader
from utils import generate_target_heatmaps
from model import DirectRegressionNet, HeatmapCornerNet


def train_dry_run(model_type="direct"):
    batch_size = 4
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Starting {model_type.upper()} dry run on {device} ---")

    
    dataset = DocumentDataset(
        clean_scans_dir='data/scan',
        backgrounds_dir='data/background',
        image_size=256,
        epoch_size=16  
    )
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
    
    for batch_idx, (degraded_imgs, _, corners) in enumerate(dataloader):
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

    print(f"Dry run complete! {model_type.upper()} architecture is valid!")

def train_model(model_type="direct", epochs=15):
    batch_size = 16
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training {model_type.upper()} on {device} ---")
    
    base_dir = '/content/drive/MyDrive/scanit_data'
    
    train_dataset = DocumentDataset(
        clean_scans_dir=f'{base_dir}/scan/train', 
        backgrounds_dir=f'{base_dir}/background', 
        image_size=256, 
        epoch_size=800
    )
    val_dataset = DocumentDataset(
        clean_scans_dir=f'{base_dir}/scan/val', 
        backgrounds_dir=f'{base_dir}/background', 
        image_size=256, 
        epoch_size=50
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Pre-generating and freezing validation set in RAM...")
    frozen_val_batches = []
    for degraded_imgs, clean_imgs, corners in val_loader:
        frozen_val_batches.append((degraded_imgs, clean_imgs, corners))
    print(f"Frozen {len(frozen_val_batches)} validation batches.")

    if model_type == "direct":
        model = DirectRegressionNet().to(device)
        criterion = nn.L1Loss()
    elif model_type == "heatmap":
        model = HeatmapCornerNet().to(device)
        criterion = nn.MSELoss()
    else:
        raise ValueError("model_type must be 'direct' or 'heatmap'")

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for degraded_imgs, _, corners in train_loader:
            degraded_imgs, corners = degraded_imgs.to(device), corners.to(device)
            targets = generate_target_heatmaps(corners) if model_type == "heatmap" else corners

            optimizer.zero_grad()
            predictions = model(degraded_imgs)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for degraded_imgs, _, corners in frozen_val_batches:
                degraded_imgs, corners = degraded_imgs.to(device), corners.to(device)
                targets = generate_target_heatmaps(corners) if model_type == "heatmap" else corners
                
                predictions = model(degraded_imgs)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(frozen_val_batches)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_type}_best.pth")
            print("🌟 New best model saved!")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title(f'{model_type.upper()} Corner Detection Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_type}_loss_curve.png')
    print(f"Loss curve saved as {model_type}_loss_curve.png\n")


if __name__ == "__main__":
    train_dry_run(model_type="direct")
    train_dry_run(model_type="heatmap")

    # train_model(model_type="direct", epochs=20)
    # train_model(model_type="heatmap", epochs=20)