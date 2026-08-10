import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import DocumentDataset
from model import DirectRegressionNet, HeatmapCornerNet
from utils import generate_target_heatmaps

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

if __name__ == "__main__":
    
    train_dry_run(model_type="direct")
    train_dry_run(model_type="heatmap")