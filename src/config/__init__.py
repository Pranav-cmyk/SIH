"""
Configuration module for Smart Grid Fault Detection System.

This module provides centralized configuration management for all components
of the fault detection system.
"""

from .settings import (
    # Paths
    PROJECT_ROOT,
    DATA_DIR,
    OUTPUT_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    
    # Dataset configuration
    DATASET_CONFIG,
    
    # Feature configuration
    FEATURE_CONFIG,
    
    # Model configuration
    MODEL_CONFIG,
    
    # Training configuration
    TRAINING_CONFIG,
    
    # Synthetic data configuration
    SYNTHETIC_CONFIG,
    
    # Evaluation configuration
    EVALUATION_CONFIG,
    
    # Logging configuration
    LOGGING_CONFIG,
)

__all__ = [
    'PROJECT_ROOT',
    'DATA_DIR',
    'OUTPUT_DIR',
    'MODELS_DIR',
    'REPORTS_DIR',
    'LOGS_DIR',
    'DATASET_CONFIG',
    'FEATURE_CONFIG',
    'MODEL_CONFIG',
    'TRAINING_CONFIG',
    'SYNTHETIC_CONFIG',
    'EVALUATION_CONFIG',
    'LOGGING_CONFIG',
]
