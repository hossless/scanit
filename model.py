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
        
        # ENCODER
        self.enc1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2) 
        
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2) 
        
        self.enc3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU())
        self.pool3 = nn.MaxPool2d(2) 
        
        # BOTTLENECK
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.ReLU()
        ) 
        
        # DECODER (With Skip Connections)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.dec3 = nn.Sequential(nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1), nn.ReLU())
        
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2) 
        self.dec2 = nn.Sequential(nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1), nn.ReLU())
        
        self.up1 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2) 
        self.dec1 = nn.Sequential(nn.Conv2d(16 + 16, 16, kernel_size=3, padding=1), nn.ReLU())
        
        # FINAL HEAD (4 Channels for the 4 corners)
        self.final_conv = nn.Conv2d(16, 4, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder Pass
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)
        
        # Bottleneck
        b = self.bottleneck(p3)
        
        # Decoder Pass with Skip Connections
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

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNetEnhancer(nn.Module):
    def __init__(self, dropout_prob=0.2):
        super(UNetEnhancer, self).__init__()
        
        # ENCODER
        self.inc = DoubleConv(3, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # BOTTLENECK
        self.bottleneck_conv = DoubleConv(256, 256)
        self.bottleneck_drop = nn.Dropout2d(dropout_prob)
        
        # DECODER
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.drop1 = nn.Dropout2d(dropout_prob)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        
        # FINAL OUTPUT LAYER
        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Bottleneck
        b = self.bottleneck_conv(x4)
        b = self.bottleneck_drop(b)
        
        # Decoder
        x = self.up1(b)
        x = torch.cat([x, x3], dim=1) 
        x = self.conv_up1(x)
        x = self.drop1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        out = self.final_conv(x)
        return self.sigmoid(out)