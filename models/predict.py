# src/model/predict.py
from models.cnn_vit_3d_model import get_model
from utils.config import config
from utils.logger import logger
import torch
import cv2
from torchvision import transforms

def predict_image(image_path: str, model_path: str = str(config.MODELS_DIR / "best_model.pth")):
    model = get_model(pretrained=False)
    model.load(model_path)
    model.to(config.DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = transform(img).unsqueeze(0).to(config.DEVICE)

    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    
    logger.info(f"Predicted class for {image_path}: {pred.item()}")
    return pred.item()

if __name__ == "__main__":
    config.ensure_directories()
    predict_image(str(config.PREPROCESSED_DIR / "sample_enhanced.jpg"))