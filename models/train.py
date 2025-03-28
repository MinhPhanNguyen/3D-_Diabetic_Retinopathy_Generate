import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import os
from datetime import datetime

from models.cnn_vit_3d_model import MultiModalRetinopathyModel
from utils.config import config
from utils.logger import logger
from utils.helpers import load_split, save_checkpoint

class DeduplicatedDataset(Dataset):
    def __init__(self, split_file):
        self.samples = self._load_deduplicated(split_file)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_deduplicated(self, split_file):
        unique_pairs = {}
        with open(split_file, 'r') as f:
            for line in f:
                img_path, label = line.strip().split(',')
                pair_id = Path(img_path).stem.split('_')[0]  # Extract '0' from '0_left.jpg'
                if pair_id not in unique_pairs:
                    unique_pairs[pair_id] = (img_path, int(label))
        return list(unique_pairs.values())
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return self.transform(img), label

class MultiModalDataset(torch.utils.data.Dataset):
    """Dataset for both 2D images and 3D models"""
    
    def __init__(self, image_files, model_files, labels):
        self.image_files = image_files
        self.model_files = model_files
        self.labels = labels
        
        # 2D transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Validate inputs
        if len(image_files) != len(model_files) or len(image_files) != len(labels):
            raise ValueError("Mismatched input lengths")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load 2D image
        img = cv2.imread(self.image_files[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        
        # Load 3D model (as point cloud)
        pcd = o3d.io.read_point_cloud(self.model_files[idx])
        points = np.asarray(pcd.points)
        
        # Sample fixed number of points
        if len(points) > config.NUM_POINTS:
            idx = np.random.choice(len(points), config.NUM_POINTS, replace=False)
            points = points[idx]
        elif len(points) < config.NUM_POINTS:
            idx = np.random.choice(len(points), config.NUM_POINTS - len(points), replace=True)
            points = np.concatenate([points, points[idx]])
        
        points = torch.from_numpy(points.T).float()  # (3, num_points)
        
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return (img, points), label

def train_epoch(model, dataloader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    
    for (images, points), labels in dataloader:
        images = images.to(config.DEVICE)
        points = points.to(config.DEVICE)
        labels = labels.to(config.DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images, points)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = correct / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    
    with torch.no_grad():
        for (images, points), labels in dataloader:
            images = images.to(config.DEVICE)
            points = points.to(config.DEVICE)
            labels = labels.to(config.DEVICE)
            
            outputs = model(images, points)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
    
    val_loss = running_loss / len(dataloader)
    val_acc = correct / len(dataloader.dataset)
    return val_loss, val_acc

def train():
    # Load data
    train_files, train_labels = load_split(config.SPLIT_DIR / "train.txt")
    val_files, val_labels = load_split(config.SPLIT_DIR / "val.txt")
    
    # Get corresponding 3D model paths
    train_models = [str(config.MODELS_3D_DIR / f"{os.path.splitext(os.path.basename(f))[0]}.ply") 
                   for f in train_files]
    val_models = [str(config.MODELS_3D_DIR / f"{os.path.splitext(os.path.basename(f))[0]}.ply") 
                 for f in val_files]
    
    # Create datasets
    train_dataset = MultiModalDataset(train_files, train_models, train_labels)
    val_dataset = MultiModalDataset(val_files, val_models, val_labels)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Initialize model
    model = MultiModalRetinopathyModel()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    
    best_acc = 0.0
    
    for epoch in range(config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion)
        
        scheduler.step(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{config.NUM_EPOCHS}: "
                   f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                   f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best=True)

    # Add to train() before training loop
    print("\n=== DATA SUMMARY ===")
    print(f"Training samples: {len(train_dataset)} (after deduplication)")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Sample shape: {train_dataset[0][0][0].shape}")
    print(f"Classes distribution: {np.bincount(train_labels)}")