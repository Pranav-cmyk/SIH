"""
Model training pipeline for Smart Grid Fault Detection.

This module orchestrates the training and evaluation of machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import logging
from typing import Dict, Any, Tuple, Optional
from pathlib import Path

from .neural_network import NeuralNetwork
from ..config import DATASET_CONFIG, EVALUATION_CONFIG, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Orchestrates model training and evaluation pipeline.
    """
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_score = 0.0
        
    def train_neural_network(self, X: pd.DataFrame, y: pd.Series, 
                           test_size: float = None) -> Dict[str, Any]:
        """
        Train neural network model.
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Fraction of data to use for testing
            
        Returns:
            Training results and metrics
        """
        if test_size is None:
            test_size = DATASET_CONFIG["test_size"]
            
        logger.info("Training Neural Network model")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=DATASET_CONFIG["random_state"], 
            stratify=y
        )
        
        # Further split training data for validation
        val_size = DATASET_CONFIG["validation_size"]
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=val_size, 
            random_state=DATASET_CONFIG["random_state"], stratify=y_train
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        
        # Initialize and train model
        nn_model = NeuralNetwork()
        training_metrics = nn_model.fit(X_train.values, y_train.values, 
                                      X_val.values, y_val.values)
        
        # Evaluate on test set
        test_metrics = nn_model.evaluate(X_test.values, y_test.values)
        
        # Make predictions for detailed analysis
        y_pred = nn_model.predict(X_test.values)
        y_pred_proba = nn_model.predict_proba(X_test.values)
        
        # Store model and results
        self.models["neural_network"] = nn_model
        self.results["neural_network"] = {
            "training_metrics": training_metrics,
            "test_metrics": test_metrics,
            "predictions": y_pred,
            "probabilities": y_pred_proba,
            "test_data": (X_test, y_test)
        }
        
        # Update best model
        if test_metrics["accuracy"] > self.best_score:
            self.best_model = nn_model
            self.best_score = test_metrics["accuracy"]
        
        logger.info(f"Neural Network training completed - Accuracy: {test_metrics['accuracy']:.4f}")
        
        return self.results["neural_network"]
    
    def get_classification_report(self, model_name: str = "neural_network") -> str:
        """
        Get detailed classification report for a model.
        
        Args:
            model_name: Name of the model to get report for
            
        Returns:
            Classification report string
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        X_test, y_test = self.results[model_name]["test_data"]
        y_pred = self.results[model_name]["predictions"]
        
        return classification_report(y_test, y_pred)
    
    def get_confusion_matrix(self, model_name: str = "neural_network") -> np.ndarray:
        """
        Get confusion matrix for a model.
        
        Args:
            model_name: Name of the model to get matrix for
            
        Returns:
            Confusion matrix
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        X_test, y_test = self.results[model_name]["test_data"]
        y_pred = self.results[model_name]["predictions"]
        
        return confusion_matrix(y_test, y_pred)
    
    def get_feature_importance(self, model_name: str = "neural_network", 
                             feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        Get feature importance for a model.
        
        Args:
            model_name: Name of the model
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importance scores
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        X_test, y_test = self.results[model_name]["test_data"]
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        return model.get_feature_importance(X_test.values, feature_names)
    
    def save_best_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the best performing model.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if self.best_model is None:
            raise ValueError("No model has been trained yet")
        
        if filepath is None:
            filepath = MODELS_DIR / "best_neural_network_model.h5"
        
        self.best_model.save_model(str(filepath))
        logger.info(f"Best model saved to {filepath}")
        
        return str(filepath)
    
    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get comparison of all trained models.
        
        Returns:
            DataFrame with model comparison metrics
        """
        if not self.results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for model_name, result in self.results.items():
            metrics = result["test_metrics"]
            comparison_data.append({
                "Model": model_name,
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1-Score": metrics["f1_score"],
                "ROC-AUC": metrics["roc_auc"]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by accuracy
        if len(comparison_df) > 0:
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
        
        return comparison_df
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training process.
        
        Returns:
            Dictionary with training summary
        """
        summary = {
            "models_trained": len(self.models),
            "best_model": "neural_network" if self.best_model else None,
            "best_accuracy": self.best_score,
            "model_comparison": self.get_model_comparison()
        }
        
        return summary
