import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from torch.utils.data import DataLoader
from utils import generate_target_heatmaps
from dataset import DocumentDataset, EnhancementDataset
from model import DirectRegressionNet, HeatmapCornerNet, UNetEnhancer

class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        kernel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.weight_x = nn.Parameter(kernel_x.repeat(3, 1, 1, 1), requires_grad=False)
        self.weight_y = nn.Parameter(kernel_y.repeat(3, 1, 1, 1), requires_grad=False)

    def forward(self, pred, target):
        pred_gx = F.conv2d(pred, self.weight_x, padding=1, groups=3)
        pred_gy = F.conv2d(pred, self.weight_y, padding=1, groups=3)
        target_gx = F.conv2d(target, self.weight_x, padding=1, groups=3)
        target_gy = F.conv2d(target, self.weight_y, padding=1, groups=3)
        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)
        return loss_x + loss_y

class DocumentEnhancementLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5):
        super(DocumentEnhancementLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1 = nn.L1Loss()
        self.sobel = SobelEdgeLoss()

    def forward(self, pred, target):
        l1_loss = self.l1(pred, target)
        ssim_val = ms_ssim(pred, target, data_range=1.0, size_average=True)
        ssim_loss = 1.0 - ssim_val
        sobel_loss = self.sobel(pred, target)
        total_loss = (self.alpha * l1_loss) + (self.beta * ssim_loss) + (self.gamma * sobel_loss)
        return total_loss

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
        
        def weighted_bce_loss(predictions, targets):
            bce = nn.functional.binary_cross_entropy(predictions, targets, reduction='none')
            weights = (targets * 10) + 1.0 
            return torch.mean(weights * bce)
            
        criterion = weighted_bce_loss
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
    
    criterion = DocumentEnhancementLoss(alpha=1.0, beta=1.0, gamma=0.5).to(device)
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