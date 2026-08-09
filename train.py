import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DocumentDataset
from model import DirectRegressionNet

def train_dry_run():
    
    batch_size = 4
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    
    dataset = DocumentDataset(
        clean_scans_dir='data/scan',
        backgrounds_dir='data/background',
        image_size=256,
        epoch_size=16  
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    
    model = DirectRegressionNet().to(device)
    
    
    criterion = nn.L1Loss() 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    
    model.train()
    print("Starting 1-epoch dry run...")
    
    for batch_idx, (degraded_imgs, _, corners) in enumerate(dataloader):
        degraded_imgs = degraded_imgs.to(device)
        corners = corners.to(device) 

        
        predictions = model(degraded_imgs)
        loss = criterion(predictions, corners)

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch [{batch_idx+1}/{len(dataloader)}] - Loss: {loss.item():.4f}")

    print("Dry run complete! Architecture and shapes are 100% valid!")

if __name__ == "__main__":
    train_dry_run()