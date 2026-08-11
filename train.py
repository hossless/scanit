import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from utils import generate_target_heatmaps
from dataset import DocumentDataset, EnhancementDataset
from model import DirectRegressionNet, HeatmapCornerNet, UNetEnhancer


def train_corner_model(model_type="direct", epochs=20):
    batch_size = 16
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training {model_type.upper()} CORNER MODEL on {device} ---")
    
    train_dataset = DocumentDataset('/content/ready_dataset/train')
    val_dataset = DocumentDataset('/content/ready_dataset/val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

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
        
        
        for degraded_imgs, corners in train_loader:
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
            for degraded_imgs, corners in val_loader:
                degraded_imgs, corners = degraded_imgs.to(device), corners.to(device)
                targets = generate_target_heatmaps(corners) if model_type == "heatmap" else corners
                
                predictions = model(degraded_imgs)
                loss = criterion(predictions, targets)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f"{model_type}_best.pth")
            print("🌟 New best corner model saved!")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title(f'{model_type.upper()} Corner Detection Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{model_type}_loss_curve.png')
    print(f"Loss curve saved as {model_type}_loss_curve.png\n")




def train_enhancement_model(epochs=20):
    batch_size = 8 
    learning_rate = 1e-4 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- Training ENHANCEMENT MODEL on {device} ---")
    
    train_dataset = EnhancementDataset('/content/ready_dataset_enhancement/train')
    val_dataset = EnhancementDataset('/content/ready_dataset_enhancement/val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = UNetEnhancer().to(device)
    
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_train_loss = 0.0
        
        for input_imgs, target_imgs in train_loader:
            input_imgs = input_imgs.to(device)
            target_imgs = target_imgs.to(device)

            optimizer.zero_grad()
            predictions = model(input_imgs)
            loss = criterion(predictions, target_imgs)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item()
            
        avg_train_loss = running_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for input_imgs, target_imgs in val_loader:
                input_imgs = input_imgs.to(device)
                target_imgs = target_imgs.to(device)
                
                predictions = model(input_imgs)
                loss = criterion(predictions, target_imgs)
                running_val_loss += loss.item()
                
        avg_val_loss = running_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "enhancer_best.pth")
            print("🌟 New best enhancer model saved!")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title('Enhancement Network Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('enhancer_loss_curve.png')
    print(f"Loss curve saved as enhancer_loss_curve.png\n")

if __name__ == "__main__":    
    # train_corner_model(model_type="direct", epochs=20)
    # train_corner_model(model_type="heatmap", epochs=20)
    
    train_enhancement_model(epochs=20)