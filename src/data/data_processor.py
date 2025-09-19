"""
Data processing module for Smart Grid Fault Detection.

This module handles loading, cleaning, and preprocessing of smart grid data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import logging

from ..config import (
    DATASET_CONFIG, 
    FEATURE_CONFIG, 
    SYNTHETIC_CONFIG,
    OUTPUT_DIR
)

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Handles data loading, preprocessing, and feature engineering for smart grid data.
    """
    
    def __init__(self):
        """Initialize the data processor."""
        self.data: Optional[pd.DataFrame] = None
        self.feature_columns: Optional[list] = None
        self.target_column: str = FEATURE_CONFIG["target_column"]
        
    def load_data(self, file_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Load smart grid data from CSV file.
        
        Args:
            file_path: Path to the CSV file. If None, uses default from config.
            
        Returns:
            Loaded DataFrame
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
            ValueError: If the data file is empty or invalid
        """
        if file_path is None:
            file_path = DATASET_CONFIG["raw_data_file"]
            
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        logger.info(f"Loading data from {file_path}")
        
        try:
            self.data = pd.read_csv(file_path)
            logger.info(f"Loaded {len(self.data)} records with {len(self.data.columns)} columns")
            
            if len(self.data) == 0:
                raise ValueError("Data file is empty")
                
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_target_variable(self) -> pd.DataFrame:
        """
        Create target variable for broken line fault detection.
        
        Returns:
            DataFrame with target variable added
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Creating target variable for broken line faults")
        
        # Initialize target variable
        self.data[self.target_column] = 0
        
        # Define fault conditions based on electrical parameters
        fault_conditions = []
        
        # Condition 1: Low voltage (potential line issues)
        if "Voltage (V)" in self.data.columns:
            threshold = SYNTHETIC_CONFIG["fault_conditions"]["low_voltage_threshold"]
            fault_conditions.append(self.data["Voltage (V)"] < threshold)
        
        # Condition 2: High voltage fluctuation (instability)
        if "Voltage Fluctuation (%)" in self.data.columns:
            threshold = SYNTHETIC_CONFIG["fault_conditions"]["high_voltage_fluctuation_threshold"]
            fault_conditions.append(self.data["Voltage Fluctuation (%)"] > threshold)
        
        # Condition 3: Overload condition (system stress)
        if "Overload Condition" in self.data.columns:
            fault_conditions.append(self.data["Overload Condition"] == 1)
        
        # Condition 4: Transformer fault (equipment failure)
        if "Transformer Fault" in self.data.columns:
            fault_conditions.append(self.data["Transformer Fault"] == 1)
        
        # Condition 5: Low power factor (poor efficiency)
        if "Power Factor" in self.data.columns:
            threshold = SYNTHETIC_CONFIG["fault_conditions"]["low_power_factor_threshold"]
            fault_conditions.append(self.data["Power Factor"] < threshold)
        
        # Condition 6: Very low current with normal voltage (open circuit)
        if all(col in self.data.columns for col in ["Voltage (V)", "Current (A)"]):
            fault_conditions.append(
                (self.data["Voltage (V)"] > 200) & (self.data["Current (A)"] < 1.0)
            )
        
        # Condition 7: Zero power consumption with predicted load (disconnection)
        if all(col in self.data.columns for col in ["Power Consumption (kW)", "Predicted Load (kW)"]):
            fault_conditions.append(
                (self.data["Power Consumption (kW)"] < 0.5) & 
                (self.data["Predicted Load (kW)"] > 2.0)
            )
        
        # Mark faults
        if fault_conditions:
            combined_condition = fault_conditions[0]
            for condition in fault_conditions[1:]:
                combined_condition = combined_condition | condition
            self.data.loc[combined_condition, self.target_column] = 1
        
        fault_count = self.data[self.target_column].sum()
        fault_percentage = (fault_count / len(self.data)) * 100
        
        logger.info(f"Detected {fault_count} fault cases ({fault_percentage:.2f}%)")
        
        return self.data
    
    def select_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select and prepare features for training.
        
        Returns:
            Tuple of (features, target)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Selecting features for training")
        
        # Combine all feature columns
        all_features = (
            FEATURE_CONFIG["numerical_features"] + 
            FEATURE_CONFIG["categorical_features"]
        )
        
        # Filter to existing columns
        self.feature_columns = [col for col in all_features if col in self.data.columns]
        
        # Prepare features and target
        X = self.data[self.feature_columns].copy()
        y = self.data[self.target_column].copy()
        
        logger.info(f"Selected {len(self.feature_columns)} features for training")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Returns:
            Cleaned DataFrame
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        logger.info("Cleaning data")
        
        # Handle missing values
        missing_before = self.data.isnull().sum().sum()
        
        # Fill numerical columns with median
        numerical_cols = FEATURE_CONFIG["numerical_features"]
        for col in numerical_cols:
            if col in self.data.columns:
                self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = FEATURE_CONFIG["categorical_features"]
        for col in categorical_cols:
            if col in self.data.columns:
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
        
        missing_after = self.data.isnull().sum().sum()
        logger.info(f"Handled {missing_before - missing_after} missing values")
        
        # Handle outliers using IQR method for numerical columns
        for col in numerical_cols:
            if col in self.data.columns:
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers_before = ((self.data[col] < lower_bound) | 
                                 (self.data[col] > upper_bound)).sum()
                
                # Cap outliers instead of removing them
                self.data[col] = np.where(self.data[col] < lower_bound, lower_bound, self.data[col])
                self.data[col] = np.where(self.data[col] > upper_bound, upper_bound, self.data[col])
                
                outliers_after = ((self.data[col] < lower_bound) | 
                                (self.data[col] > upper_bound)).sum()
                
                if outliers_before > 0:
                    logger.info(f"Capped {outliers_before} outliers in {col}")
        
        return self.data
    
    def save_processed_data(self, file_path: Optional[Path] = None) -> Path:
        """
        Save processed data to CSV file.
        
        Args:
            file_path: Path to save the processed data
            
        Returns:
            Path where data was saved
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if file_path is None:
            file_path = DATASET_CONFIG["processed_data_file"]
            
        logger.info(f"Saving processed data to {file_path}")
        self.data.to_csv(file_path, index=False)
        
        return file_path
    
    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dictionary containing data summary
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        summary = {
            "total_records": len(self.data),
            "total_features": len(self.data.columns),
            "fault_cases": self.data[self.target_column].sum() if self.target_column in self.data.columns else 0,
            "fault_percentage": (self.data[self.target_column].sum() / len(self.data) * 100) if self.target_column in self.data.columns else 0,
            "missing_values": self.data.isnull().sum().sum(),
            "feature_columns": self.feature_columns,
            "data_types": self.data.dtypes.to_dict()
        }
        
        return summary
