import os
from pathlib import Path

class Config:
    """
        Configuration class for 3D Diabetic Retinopathy project
    """
    
    BASE_DIR = Path(__file__).parent.parent.parent
    
    # Data directories
    DATA_DIR = BASE_DIR / "3D_Stimualation/data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PREPROCESSED_DIR = DATA_DIR / "preprocessed"
    PAIR_PREPROCESSED_DIR = DATA_DIR / "pair_preprocessed"
    MODEL_REFINED_DIR = DATA_DIR / "refined_models"
    SYNTHETIC_DIR = DATA_DIR / "synthetic"
    DEPTH_MAPS_DIR = DATA_DIR / "depth_maps"
    MODELS_3D_DIR = DATA_DIR / "3D_models"
    ANNOTATIONS_DIR = DATA_DIR / "annotations"
    SPLIT_DIR = DATA_DIR / "split"
    
    # Output directories
    OUTPUT_DIR = BASE_DIR / "output"
    PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
    MODELS_DIR = OUTPUT_DIR / "models"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    LOGS_DIR = OUTPUT_DIR / "logs"
    
    # Model hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    NUM_CLASSES = 5  
    
    # Preprocessing parameters
    CONTRAST_FACTOR = 1.5
    DENOISE_STRENGTH = 10
    IMAGE_SIZE = (384, 384) 
    
    # Reconstruction parameters
    DEPTH_SCALE = 1000.0
    
    # Device configuration
    DEVICE = "cuda" if os.environ.get("CUDA_AVAILABLE", "False") == "True" else "cpu"
    
    # Logging
    LOG_LEVEL = "INFO"

    @staticmethod
    def ensure_directories():
        """
            Create all necessary directories if they don't exist
        """
        for attr in dir(Config):
            if attr.endswith('_DIR') and isinstance(getattr(Config, attr), Path):
                getattr(Config, attr).mkdir(parents=True, exist_ok=True)

# Singleton instance
config = Config()

if __name__ == "__main__":
    # Test the config
    config.ensure_directories()
    print(f"Project root: {config.BASE_DIR}")
    print(f"Device: {config.DEVICE}")