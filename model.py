import torch
import torch.nn as nn

class BasicYOLODetector(nn.Module):
    def __init__(self, in_channels=3, S=7, B=1, C=2):
        super(BasicYOLODetector, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        # --- Simplified CNN Backbone (UNCHANGED LAYERS) ---
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        # --- Detection Head (FIXED INPUT SIZE) ---
        # The input size must match the actual output of the backbone: 256 * H * W
        # Assuming IMG_SIZE=224 was used, H*W = 14*14
        FINAL_FEATURE_SIZE = 256 * 14 * 14 # = 50176 (MATCHES ERROR TRACE)
        OUTPUT_SIZE = self.S * self.S * (self.C + 5 * self.B)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(FINAL_FEATURE_SIZE, 4096), # FIXED: Uses 50176 as input size
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, OUTPUT_SIZE)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        
        return x.reshape(-1, self.S, self.S, self.C + 5 * self.B)
