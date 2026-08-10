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
    

class HeatmapCornerNet(nn.Module):
    def __init__(self):
        super(HeatmapCornerNet, self).__init__()
        
        # ENCODER (Shrinks the image)
        # Input: (Batch, 3, 256, 256)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2) # Outputs: (16, 128, 128)
        
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2) # Outputs: (32, 64, 64)
        
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2) # Outputs: (64, 32, 32)
        
        self.enc4 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.pool4 = nn.MaxPool2d(2) # Outputs: (128, 16, 16)
        
        # DECODER (Upsamples back to 256x256)
        # ConvTranspose2d doubles the spatial dimensions
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # (64, 32, 32)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # (32, 64, 64)
        self.dec2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU())
        
        self.up3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) # (16, 128, 128)
        self.dec3 = nn.Sequential(nn.Conv2d(16, 16, kernel_size=3, padding=1), nn.ReLU())
        
        self.up4 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2) # (16, 256, 256)
        
        # FINAL HEAD
        # Output is exactly 4 channels (one for each corner mask)
        self.final_conv = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder Pass
        e1 = self.enc1(x)
        x = self.pool1(e1)
        
        e2 = self.enc2(x)
        x = self.pool2(e2)
        
        e3 = self.enc3(x)
        x = self.pool3(e3)
        
        e4 = self.enc4(x)
        x = self.pool4(e4)
        
        # Decoder Pass
        x = self.up1(x)
        x = self.dec1(x)
        
        x = self.up2(x)
        x = self.dec2(x)
        
        x = self.up3(x)
        x = self.dec3(x)
        
        x = self.up4(x)
        
        # Output exactly (Batch, 4, 256, 256)
        x = self.final_conv(x)
        return self.sigmoid(x)