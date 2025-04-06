"""Contains the CNN model architecture."""

import torch
import torch.nn as nn
import torch.nn.functional as F

# === CNN Model ===
class CNN_From_Diagram(nn.Module):
    """
    CNN architecture based on a standard VGG-like pattern (Conv-Conv-Pool blocks).
    Designed for CIFAR-10 (32x32 input). 6 Conv, 3 MaxPool, 1 Hidden FC, 1 Output FC.
    Includes BatchNorm, lacks Dropout. Matches Nagarsekar report Appendix A.1.
    """
    def __init__(self, num_classes=10):
        super(CNN_From_Diagram, self).__init__()

        # --- Block 1 --- (Input: B x 3 x 32 x 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B x 64 x 16 x 16

        # --- Block 2 --- (Input: B x 64 x 16 x 16)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B x 128 x 8 x 8

        # --- Block 3 --- (Input: B x 128 x 8 x 8)
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: B x 256 x 4 x 4

        # --- Classifier Head ---
        # Calculate the flattened feature size after the last pooling layer
        self.flattened_size = 256 * 4 * 4 # = 4096

        # Fully Connected Layer 1 (Hidden Layer)
        self.fc1 = nn.Linear(self.flattened_size, 512)
        # Fully Connected Layer 2 (Output Layer)
        self.fc2 = nn.Linear(512, num_classes)
        # Design Choice: No Dropout layers included, following the reference diagram strictly.

    def forward(self, x):
        """Defines the forward pass of the CNN."""
        # Block 1 Pass: Conv -> BN -> ReLU -> Conv -> BN -> ReLU -> Pool
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))

        # Block 2 Pass
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))

        # Block 3 Pass
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool3(F.relu(self.bn6(self.conv6(x))))

        # Flatten the output for the classifier
        x = x.view(-1, self.flattened_size) # Reshape (B, C, H, W) -> (B, C*H*W)

        # Classifier Pass
        x = F.relu(self.fc1(x)) # Hidden layer with ReLU activation
        x = self.fc2(x)         # Output layer (returns logits)

        return x