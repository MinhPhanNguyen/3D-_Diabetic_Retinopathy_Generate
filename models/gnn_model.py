# models/gnn_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import open3d as o3d
from utils.config import config
from utils.logger import logger

class PointNetClassifier(nn.Module):
    """Simple PointNet for 3D point cloud classification"""
    
    def __init__(self, num_classes=config.NUM_CLASSES, num_points=1024):
        super(PointNetClassifier, self).__init__()
        self.num_points = num_points
        
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        self.dropout = nn.Dropout(0.5)
        
        self.to(config.DEVICE)
        logger.info(f"PointNet initialized with {num_classes} classes")

    def forward(self, x):
        # x: (batch_size, 3, num_points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        
        x = torch.max(x, 2, keepdim=True)[0]  # Global max pooling
        x = x.view(-1, 1024)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

    def save(self, path: str):
        torch.save(self.state_dict(), path)
        logger.info(f"Model saved to: {path}")

    def load(self, path: str):
        state_dict = torch.load(path, map_location=config.DEVICE)
        self.load_state_dict(state_dict)
        self.to(config.DEVICE)
        logger.info(f"Model loaded from: {path}")

def load_point_cloud(file_path: str, num_points: int = 1024) -> torch.Tensor:
    """Load and preprocess point cloud"""
    pcd = o3d.io.read_point_cloud(file_path)
    points = np.asarray(pcd.points)
    
    if len(points) > num_points:
        idx = np.random.choice(len(points), num_points, replace=False)
        points = points[idx]
    elif len(points) < num_points:
        idx = np.random.choice(len(points), num_points - len(points), replace=True)
        points = np.concatenate([points, points[idx]])
    
    return torch.tensor(points, dtype=torch.float32).transpose(0, 1)  # (3, num_points)

if __name__ == "__main__":
    model = PointNetClassifier()
    sample = load_point_cloud("D:\\3D_Stimualation\\data\\3D_models\\0_left.ply")
    sample = sample.unsqueeze(0).to(config.DEVICE)  # (1, 3, 1024)
    output = model(sample)
    logger.info(f"Output shape: {output.shape}")