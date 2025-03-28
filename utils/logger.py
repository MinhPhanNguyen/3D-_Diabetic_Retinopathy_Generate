# src/utils/logger.py
import os
import logging
from datetime import datetime
from typing import Optional
from utils.config import config  # Import config for paths and settings

class Logger:
    """Custom logging class for the 3D Diabetic Retinopathy project"""
    
    def __init__(self, 
                 name: str = "3D_Diabetic_Retinopathy",
                 log_level: str = None,
                 log_to_file: bool = True):
        """
        Initialize the logger
        
        Args:
            name: Name of the logger
            log_level: Logging level (e.g., "INFO", "DEBUG"). Defaults to config.LOG_LEVEL
            log_to_file: Whether to log to a file in addition to console
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Base level, handlers will filter
        
        # Avoid duplicate handlers if logger is reinitialized
        if not self.logger.handlers:
            # Set log level from parameter or config
            log_level = log_level or config.LOG_LEVEL
            numeric_level = getattr(logging, log_level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level: {log_level}")
            
            # Formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(numeric_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
            
            # File handler (optional)
            if log_to_file:
                log_dir = config.LOGS_DIR
                log_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = log_dir / f"run_{timestamp}.log"
                
                file_handler = logging.FileHandler(log_file)
                file_handler.setLevel(numeric_level)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
                
                self.logger.info(f"Logging to file: {log_file}")

    def get_logger(self) -> logging.Logger:
        """Return the configured logger instance"""
        return self.logger

    def debug(self, msg: str) -> None:
        """Log a debug message"""
        self.logger.debug(msg)

    def info(self, msg: str) -> None:
        """Log an info message"""
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        """Log a warning message"""
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log an error message"""
        self.logger.error(msg)

    def critical(self, msg: str) -> None:
        """Log a critical message"""
        self.logger.critical(msg)

    @staticmethod
    def setup_global_logger(name: str = "3D_Diabetic_Retinopathy",
                          log_level: Optional[str] = None) -> logging.Logger:
        """
        Setup and return a global logger instance
        
        Args:
            name: Logger name
            log_level: Override default log level from config
        """
        logger_instance = Logger(name=name, log_level=log_level)
        return logger_instance.get_logger()

# Singleton logger instance
logger = Logger().get_logger()

if __name__ == "__main__":
    # Test the logger
    test_logger = Logger(log_level="DEBUG")
    log = test_logger.get_logger()
    
    log.debug("This is a debug message")
    log.info("This is an info message")
    log.warning("This is a warning message")
    log.error("This is an error message")
    log.critical("This is a critical message")
    
    # Test global logger
    global_log = Logger.setup_global_logger()
    global_log.info("Global logger is working")