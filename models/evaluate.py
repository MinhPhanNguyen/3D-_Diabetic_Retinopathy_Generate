# src/model/evaluate.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from models.cnn_vit_3d_model import get_model
from models.train import RetinopathyDataset, load_split
from utils.config import config
from utils.logger import logger
from utils.visualization import Visualizer

def evaluate_model(model_path: str = str(config.MODELS_DIR / "best_model.pth"),
                  test_split_file: str = str(config.SPLIT_DIR / "test.txt")):
    """Evaluate the trained model on a test set"""
    
    # Load test data
    if not torch.cuda.is_available() and config.DEVICE == "cuda":
        logger.warning("CUDA not available, using CPU")
        config.DEVICE = "cpu"
    
    if not os.path.exists(test_split_file):
        logger.error(f"Test split file not found: {test_split_file}")
        raise FileNotFoundError("Test split file not found. Create it with helpers.py")
    
    test_images, test_labels = load_split(test_split_file)
    if not test_images:
        logger.error("No test data found in split file")
        raise ValueError("Test split file is empty")
    
    test_dataset = RetinopathyDataset(test_images, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Load model
    model = get_model(pretrained=False)  # No pretrained weights, we'll load our own
    try:
        model.load(model_path)
        model.to(config.DEVICE)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model from {model_path}: {str(e)}")
        raise
    
    # Evaluation
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Metrics
    accuracy = accuracy_score(all_labels, all_preds)
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Visualize confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(config.NUM_CLASSES),
                yticklabels=range(config.NUM_CLASSES))
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    conf_matrix_path = str(config.VISUALIZATIONS_DIR / "confusion_matrix.png")
    plt.savefig(conf_matrix_path)
    logger.info(f"Confusion matrix saved to: {conf_matrix_path}")
    plt.close()

if __name__ == "__main__":
    import os
    config.ensure_directories()
    evaluate_model()