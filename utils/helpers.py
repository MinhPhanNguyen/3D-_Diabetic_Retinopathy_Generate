# src/utils/helpers.py
import os
import numpy as np
import cv2
from typing import List, Tuple, Union, Optional
from pathlib import Path
from utils.config import config
from utils.logger import logger

class Helpers:
    """Utility functions for the 3D Diabetic Retinopathy project"""

    @staticmethod
    def list_files(directory: Union[str, Path], 
                  extensions: List[str] = [".jpg", ".png"]) -> List[str]:
        """
        List all files in a directory with specified extensions
        
        Args:
            directory: Directory path to search
            extensions: List of file extensions to filter (e.g., [".jpg", ".png"])
        
        Returns:
            List of file paths
        """
        directory = Path(directory)
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            raise FileNotFoundError(f"Directory not found: {directory}")
            
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files)} files in {directory}")
        return [str(f) for f in files]

    @staticmethod
    def normalize_array(arr: np.ndarray, 
                      min_val: float = 0.0, 
                      max_val: float = 1.0) -> np.ndarray:
        """
        Normalize a numpy array to a specified range
        
        Args:
            arr: Input array
            min_val: Minimum value of output range
            max_val: Maximum value of output range
        
        Returns:
            Normalized array
        """
        arr_min, arr_max = np.min(arr), np.max(arr)
        if arr_max == arr_min:
            logger.warning("Array has no range, returning zeros")
            return np.zeros_like(arr)
            
        normalized = (arr - arr_min) / (arr_max - arr_min) * (max_val - min_val) + min_val
        return normalized

    @staticmethod
    def resize_image(image_path: str, 
                    size: Tuple[int, int] = config.IMAGE_SIZE,
                    output_path: Optional[str] = None) -> np.ndarray:
        """
        Resize an image to specified dimensions
        
        Args:
            image_path: Path to input image
            size: Target size (height, width)
            output_path: Optional path to save resized image
        
        Returns:
            Resized image as numpy array
        """
        if not os.path.exists(image_path):
            logger.error(f"Image not found: {image_path}")
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img = cv2.imread(image_path)
        resized = cv2.resize(img, size[::-1], interpolation=cv2.INTER_AREA)
        
        if output_path:
            cv2.imwrite(output_path, resized)
            logger.info(f"Resized image saved to: {output_path}")
            
        return resized

    @staticmethod
    def split_dataset(file_list: List[str], 
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15) -> Tuple[List[str], List[str], List[str]]:
        """
        Split a list of files into train, validation, and test sets
        
        Args:
            file_list: List of file paths
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
        
        Returns:
            Tuple of (train_files, val_files, test_files)
        """
        if not 0 < train_ratio + val_ratio < 1:
            logger.error("Invalid split ratios: train + val must be between 0 and 1")
            raise ValueError("Invalid split ratios")
            
        np.random.shuffle(file_list)
        n_total = len(file_list)
        
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        train_files = file_list[:n_train]
        val_files = file_list[n_train:n_train + n_val]
        test_files = file_list[n_train + n_val:]
        
        logger.info(f"Dataset split: {len(train_files)} train, "
                   f"{len(val_files)} val, {len(test_files)} test")
        return train_files, val_files, test_files

    @staticmethod
    def save_split_to_files(train_files: List[str], 
                          val_files: List[str], 
                          test_files: List[str],
                          output_dir: Union[str, Path] = config.SPLIT_DIR) -> None:
        """
        Save dataset split to text files
        
        Args:
            train_files: List of training file paths
            val_files: List of validation file paths
            test_files: List of test file paths
            output_dir: Directory to save split files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split_name, file_list in [("train.txt", train_files), 
                                    ("val.txt", val_files), 
                                    ("test.txt", test_files)]:
            with open(output_dir / split_name, 'w') as f:
                f.write("\n".join(file_list))
            logger.info(f"Saved {split_name} with {len(file_list)} entries to {output_dir}")

if __name__ == "__main__":
    # Test the helper functions
    helpers = Helpers()
    
    # Test file listing
    files = helpers.list_files(config.RAW_DATA_DIR)
    print(f"Found {len(files)} files")
    
    # Test normalization
    sample_array = np.random.rand(10, 10) * 100
    normalized = helpers.normalize_array(sample_array)
    print(f"Normalized array min: {normalized.min()}, max: {normalized.max()}")
    
    # Test image resizing (assuming a sample image exists)
    if files:
        resized = helpers.resize_image(files[0], size=(224, 224))
        print(f"Resized image shape: {resized.shape}")
    
    # Test dataset splitting
    if files:
        train, val, test = helpers.split_dataset(files)
        helpers.save_split_to_files(train, val, test)
        print(f"Split sizes: train={len(train)}, val={len(val)}, test={len(test)}")