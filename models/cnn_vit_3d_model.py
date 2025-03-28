import torch
import torch.nn as nn
from torchvision import models
from typing import Optional, Tuple
from utils.config import config
from utils.logger import logger

class MultiModalRetinopathyModel(nn.Module):
    """Unified 2D-3D model for diabetic retinopathy classification"""
    
    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()
        
        # 2D Image Pathway (CNN-ViT)
        self.cnn = models.resnet50(pretrained=True)
        self.cnn.fc = nn.Identity()  # Remove final layer
        
        self.vit = models.vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()  # Remove classification head
        
        # 3D Pathway (PointNet-like)
        self.pointnet_conv1 = nn.Conv1d(3, 64, 1)
        self.pointnet_conv2 = nn.Conv1d(64, 128, 1)
        self.pointnet_conv3 = nn.Conv1d(128, 1024, 1)
        self.pointnet_bn1 = nn.BatchNorm1d(64)
        self.pointnet_bn2 = nn.BatchNorm1d(128)
        self.pointnet_bn3 = nn.BatchNorm1d(1024)
        
        # Fusion and Classification
        self.fusion = nn.Linear(2048 + 768 + 1024, 1024)  # ResNet50 + ViT + PointNet
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        self.to(config.DEVICE)
        logger.info("Initialized MultiModalRetinopathyModel")

    def forward_2d(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process 2D image through both CNN and ViT"""
        cnn_features = self.cnn(x)  # (batch_size, 2048)
        vit_features = self.vit(x)  # (batch_size, 768)
        return cnn_features, vit_features

    def forward_3d(self, x: torch.Tensor) -> torch.Tensor:
        """Process 3D point cloud"""
        # x shape: (batch_size, 3, num_points)
        x = torch.relu(self.pointnet_bn1(self.pointnet_conv1(x)))
        x = torch.relu(self.pointnet_bn2(self.pointnet_conv2(x)))
        x = self.pointnet_bn3(self.pointnet_conv3(x))
        x = torch.max(x, 2)[0]  # Global max pooling
        return x  # (batch_size, 1024)

    def forward(self, x_2d: torch.Tensor, x_3d: torch.Tensor) -> torch.Tensor:
        """Forward pass with both 2D and 3D data"""
        cnn_feat, vit_feat = self.forward_2d(x_2d)
        point_feat = self.forward_3d(x_3d)
        
        # Concatenate all features
        combined = torch.cat([cnn_feat, vit_feat, point_feat], dim=1)
        return self.classifier(self.fusion(combined))

    def save(self, path: str) -> None:
        """Save model state"""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'num_classes': self.classifier[-1].out_features
            }
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load model state"""
        checkpoint = torch.load(path, map_location=config.DEVICE)
        self.load_state_dict(checkpoint['state_dict'])
        self.to(config.DEVICE)
        logger.info(f"Model loaded from {path}")