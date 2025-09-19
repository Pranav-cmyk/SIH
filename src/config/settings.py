"""
Configuration settings for Smart Grid Fault Detection System.

This module contains all configuration parameters organized by component.
"""

from pathlib import Path
from typing import Dict, List, Any

# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "assets"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
LOGS_DIR = PROJECT_ROOT / "logs"

# Create directories if they don't exist
for directory in [OUTPUT_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

DATASET_CONFIG: Dict[str, Any] = {
    "raw_data_file": DATA_DIR / "smart_grid_dataset.csv",
    "processed_data_file": OUTPUT_DIR / "processed_dataset.csv",
    "target_column": "Broken_Line_Fault",
    "test_size": 0.2,
    "validation_size": 0.2,
    "random_state": 42,
}

# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

FEATURE_CONFIG: Dict[str, List[str]] = {
    "numerical_features": [
        "Voltage (V)",
        "Current (A)", 
        "Power Consumption (kW)",
        "Reactive Power (kVAR)",
        "Power Factor",
        "Solar Power (kW)",
        "Wind Power (kW)",
        "Grid Supply (kW)",
        "Voltage Fluctuation (%)",
        "Temperature (°C)",
        "Humidity (%)",
        "Electricity Price (USD/kWh)"
    ],
    "categorical_features": [
        "Overload Condition",
        "Transformer Fault"
    ],
    "target_column": "Broken_Line_Fault"
}

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

MODEL_CONFIG: Dict[str, Any] = {
    "neural_network": {
        "hidden_layers": [128, 64, 32],
        "activation": "relu",
        "output_activation": "sigmoid",
        "dropout_rate": 0.3,
        "learning_rate": 0.001,
        "optimizer": "adam",
        "loss": "binary_crossentropy",
        "metrics": ["accuracy", "precision", "recall"]
    }
}

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================

TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": 32,
    "epochs": 40,
    "validation_split": 0.2,
    "early_stopping": {
        "monitor": "val_loss",
        "patience": 10,
        "restore_best_weights": True
    },
    "reduce_lr": {
        "monitor": "val_loss",
        "factor": 0.5,
        "patience": 5,
        "min_lr": 1e-7
    },
    "callbacks": ["early_stopping", "reduce_lr"]
}

# =============================================================================
# SYNTHETIC DATA CONFIGURATION
# =============================================================================

SYNTHETIC_CONFIG: Dict[str, Any] = {
    "n_synthetic_samples": 2000,
    "fault_scenarios": {
        "voltage_drop_range": (0.1, 0.5),
        "current_reduction_range": (0.05, 0.3),
        "power_factor_degradation_range": (0.3, 0.7),
        "voltage_fluctuation_multiplier_range": (3, 8),
        "power_consumption_reduction_range": (0.1, 0.4),
        "noise_level": 0.05
    },
    "fault_conditions": {
        "low_voltage_threshold": 150,
        "high_voltage_fluctuation_threshold": 3,
        "low_power_factor_threshold": 0.8
    }
}

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

EVALUATION_CONFIG: Dict[str, Any] = {
    "metrics": [
        "accuracy",
        "precision", 
        "recall",
        "f1_score",
        "roc_auc"
    ],
    "classification_threshold": 0.5,
    "cross_validation_folds": 5
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

LOGGING_CONFIG: Dict[str, Any] = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": LOGS_DIR / "fault_detection.log",
    "max_file_size": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5
}
