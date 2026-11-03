from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
import torch




class YourCNN(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        ## TODO implement
        self.dropout = dropout
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2)
        )

        self.neural_net = nn.Sequential(
            nn.Flatten(), # 128C * 4H * 4W
            nn.Linear(128 * 4 * 4, 512), 
            nn.ReLU(), 
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 256), 
            nn.ReLU(), 
            nn.Linear(256, 10)
        )

    def forward(self, x):
        ## TODO implement
        x = self.conv_block(x)
        x = self.neural_net(x)
        return x

    