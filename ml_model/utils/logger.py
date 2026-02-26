"""
Logger Configuration
Sets up centralized logging for the application
"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

def setup_logger(name, log_level=logging.INFO, log_file=None):
    """
    Configure and return a logger instance
    
    Args:
        name: Logger name (usually __name__)
        log_level: Logging level
        log_file: Path to log file (optional)
    
    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.stream.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    # Format
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)-8s %(name)-20s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10485760,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
