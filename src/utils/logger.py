"""
Logging utilities for Smart Grid Fault Detection System.

This module provides centralized logging configuration and utilities.
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from ..config import LOGGING_CONFIG


def setup_logging(log_file: Optional[Path] = None, 
                 log_level: Optional[str] = None) -> logging.Logger:
    """
    Set up logging configuration for the application.
    
    Args:
        log_file: Path to log file. If None, uses default from config.
        log_level: Logging level. If None, uses default from config.
        
    Returns:
        Configured logger instance
    """
    # Get configuration
    config = LOGGING_CONFIG
    log_file = log_file or config["file"]
    log_level = log_level or config["level"]
    
    # Create logs directory if it doesn't exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("smart_grid_fault_detection")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config["format"])
    
    # File handler with rotation
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=config["max_file_size"],
        backupCount=config["backup_count"]
    )
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Name of the module/component
        
    Returns:
        Logger instance
    """
    return logging.getLogger(f"smart_grid_fault_detection.{name}")
