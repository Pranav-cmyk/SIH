"""
Synthetic data generation module for Smart Grid Fault Detection.

This module generates realistic synthetic fault data to augment the training dataset.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

from ..config import SYNTHETIC_CONFIG, FEATURE_CONFIG

logger = logging.getLogger(__name__)


class SyntheticDataGenerator:
    """
    Generates synthetic fault data for training augmentation.
    """
    
    def __init__(self):
        """Initialize the synthetic data generator."""
        self.config = SYNTHETIC_CONFIG
        
    def generate_synthetic_faults(self, X: pd.DataFrame, y: pd.Series, 
                                n_synthetic: Optional[int] = None) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate synthetic fault data based on normal samples.
        
        Args:
            X: Feature matrix
            y: Target vector
            n_synthetic: Number of synthetic samples to generate
            
        Returns:
            Tuple of (augmented_features, augmented_targets)
        """
        if n_synthetic is None:
            n_synthetic = self.config["n_synthetic_samples"]
            
        logger.info(f"Generating {n_synthetic} synthetic fault samples")
        
        # Get normal samples (non-fault cases)
        normal_mask = y == 0
        normal_samples = X[normal_mask].copy()
        
        if len(normal_samples) == 0:
            logger.warning("No normal samples found for synthetic generation")
            return X, y
        
        synthetic_samples = []
        
        for i in range(n_synthetic):
            # Randomly select a normal sample as base
            base_sample = normal_samples.sample(1).iloc[0].copy()
            
            # Apply fault modifications
            modified_sample = self._apply_fault_modifications(base_sample)
            synthetic_samples.append(modified_sample)
        
        # Create synthetic DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples)
        synthetic_targets = pd.Series([1] * n_synthetic)  # All are faults
        
        # Combine with original data
        X_augmented = pd.concat([X, synthetic_df], ignore_index=True)
        y_augmented = pd.concat([y, synthetic_targets], ignore_index=True)
        
        logger.info(f"Augmented dataset: {len(X_augmented)} samples")
        logger.info(f"New target distribution: {y_augmented.value_counts().to_dict()}")
        
        return X_augmented, y_augmented
    
    def _apply_fault_modifications(self, sample: pd.Series) -> pd.Series:
        """
        Apply fault-specific modifications to a normal sample.
        
        Args:
            sample: Normal sample to modify
            
        Returns:
            Modified sample with fault characteristics
        """
        modified_sample = sample.copy()
        scenarios = self.config["fault_scenarios"]
        
        # Apply voltage drop (most common fault symptom)
        if "Voltage (V)" in modified_sample.index:
            voltage_range = scenarios["voltage_drop_range"]
            multiplier = np.random.uniform(voltage_range[0], voltage_range[1])
            modified_sample["Voltage (V)"] *= multiplier
        
        # Apply current reduction (open circuit effect)
        if "Current (A)" in modified_sample.index:
            current_range = scenarios["current_reduction_range"]
            multiplier = np.random.uniform(current_range[0], current_range[1])
            modified_sample["Current (A)"] *= multiplier
        
        # Degrade power factor (poor efficiency)
        if "Power Factor" in modified_sample.index:
            pf_range = scenarios["power_factor_degradation_range"]
            multiplier = np.random.uniform(pf_range[0], pf_range[1])
            modified_sample["Power Factor"] *= multiplier
        
        # Increase voltage fluctuation (instability)
        if "Voltage Fluctuation (%)" in modified_sample.index:
            fluctuation_range = scenarios["voltage_fluctuation_multiplier_range"]
            multiplier = np.random.uniform(fluctuation_range[0], fluctuation_range[1])
            modified_sample["Voltage Fluctuation (%)"] *= multiplier
        
        # Reduce power consumption (load disconnection)
        if "Power Consumption (kW)" in modified_sample.index:
            consumption_range = scenarios["power_consumption_reduction_range"]
            multiplier = np.random.uniform(consumption_range[0], consumption_range[1])
            modified_sample["Power Consumption (kW)"] *= multiplier
        
        # Add realistic noise
        noise_level = scenarios["noise_level"]
        for col in modified_sample.index:
            if modified_sample[col] != 0 and col in FEATURE_CONFIG["numerical_features"]:
                noise = np.random.normal(0, noise_level * abs(modified_sample[col]))
                modified_sample[col] += noise
        
        return modified_sample
    
    def generate_specific_fault_scenarios(self, X: pd.DataFrame, y: pd.Series,
                                        scenario_type: str = "mixed") -> Tuple[pd.DataFrame, pd.Series]:
        """
        Generate specific types of fault scenarios.
        
        Args:
            X: Feature matrix
            y: Target vector
            scenario_type: Type of fault scenario ("voltage_drop", "current_loss", "mixed")
            
        Returns:
            Tuple of (augmented_features, augmented_targets)
        """
        logger.info(f"Generating {scenario_type} fault scenarios")
        
        normal_mask = y == 0
        normal_samples = X[normal_mask].copy()
        
        if len(normal_samples) == 0:
            logger.warning("No normal samples found for scenario generation")
            return X, y
        
        n_synthetic = self.config["n_synthetic_samples"]
        synthetic_samples = []
        
        for i in range(n_synthetic):
            base_sample = normal_samples.sample(1).iloc[0].copy()
            
            if scenario_type == "voltage_drop":
                modified_sample = self._create_voltage_drop_fault(base_sample)
            elif scenario_type == "current_loss":
                modified_sample = self._create_current_loss_fault(base_sample)
            else:  # mixed
                modified_sample = self._apply_fault_modifications(base_sample)
            
            synthetic_samples.append(modified_sample)
        
        # Create synthetic DataFrame
        synthetic_df = pd.DataFrame(synthetic_samples)
        synthetic_targets = pd.Series([1] * n_synthetic)
        
        # Combine with original data
        X_augmented = pd.concat([X, synthetic_df], ignore_index=True)
        y_augmented = pd.concat([y, synthetic_targets], ignore_index=True)
        
        return X_augmented, y_augmented
    
    def _create_voltage_drop_fault(self, sample: pd.Series) -> pd.Series:
        """Create a voltage drop fault scenario."""
        modified_sample = sample.copy()
        
        # Significant voltage drop
        if "Voltage (V)" in modified_sample.index:
            modified_sample["Voltage (V)"] *= np.random.uniform(0.1, 0.4)
        
        # Reduced current
        if "Current (A)" in modified_sample.index:
            modified_sample["Current (A)"] *= np.random.uniform(0.2, 0.6)
        
        # Poor power factor
        if "Power Factor" in modified_sample.index:
            modified_sample["Power Factor"] *= np.random.uniform(0.4, 0.7)
        
        return modified_sample
    
    def _create_current_loss_fault(self, sample: pd.Series) -> pd.Series:
        """Create a current loss fault scenario."""
        modified_sample = sample.copy()
        
        # Very low current (open circuit)
        if "Current (A)" in modified_sample.index:
            modified_sample["Current (A)"] *= np.random.uniform(0.05, 0.2)
        
        # Normal voltage but no current flow
        if "Voltage (V)" in modified_sample.index:
            modified_sample["Voltage (V)"] *= np.random.uniform(0.8, 1.2)
        
        # Zero power consumption
        if "Power Consumption (kW)" in modified_sample.index:
            modified_sample["Power Consumption (kW)"] *= np.random.uniform(0.01, 0.1)
        
        return modified_sample
