import torch
import torch.nn as nn

class DirectRegressionNet(nn.Module):
    def __init__(self):
        super(DirectRegressionNet, self).__init__()
        
        self.encoder = nn.Sequential(
            # Layer 1: Input (3, 256, 256) -> Output (16, 128, 128)
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Layer 2: Output (32, 64, 64)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 3: Output (64, 32, 32)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 4: Output (128, 16, 16)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Layer 5: Output (256, 8, 8)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),

            nn.Linear(512, 8),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)
        coordinates = self.fc_head(features)
        return coordinates