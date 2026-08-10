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
    

class UNetEnhancer(nn.Module):
    def __init__(self):
        super(UNetEnhancer, self).__init__()
        
        # ENCODER (Downsampling & Feature Extraction)
        # Input: (Batch, 3, 256, 256)
        self.enc1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2) # Output: (32, 128, 128)
        
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2) # Output: (64, 64, 64)

        self.enc3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2) # Output: (128, 32, 32)
        
        # THE BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), 
            nn.ReLU()
        ) # Output: (256, 32, 32)
        
        # DECODER (Upsampling & Skip Connections)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # Back to 64x64
        self.dec3 = nn.Sequential(nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # Back to 128x128
        self.dec2 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) # Back to 256x256
        self.dec1 = nn.Sequential(nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1), nn.ReLU())
        
        # FINAL OUTPUT LAYER
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        # Forward pass through Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Pass through Bottleneck
        b = self.bottleneck(p3)
        
        # Forward pass through Decoder with Skip Connections (torch.cat)
        d3 = self.up3(b)
        d3 = torch.cat((d3, e3), dim=1) 
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1(d1)
        
        out = self.final_conv(d1)
        return self.sigmoid(out)