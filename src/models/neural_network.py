"""
Neural Network model for Smart Grid Fault Detection.

This module implements a gradient descent-based neural network for binary classification.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional
import time

from ..config import MODEL_CONFIG, TRAINING_CONFIG

logger = logging.getLogger(__name__)


class NeuralNetwork:
    """
    Gradient Descent-based Neural Network for Smart Grid Fault Detection.
    
    This class provides a complete neural network implementation with:
    - Configurable architecture
    - Early stopping and learning rate reduction
    - Comprehensive evaluation metrics
    - Model persistence
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the neural network.
        
        Args:
            config: Model configuration dictionary. If None, uses default config.
        """
        self.config = config or MODEL_CONFIG["neural_network"]
        self.model: Optional[Sequential] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.history: Optional[tf.keras.callbacks.History] = None
        self.training_config = TRAINING_CONFIG
        
        # Set random seeds for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        
        logger.info("Neural Network initialized")
    
    def _build_model(self, input_dim: int) -> Sequential:
        """
        Build the neural network architecture.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras Sequential model
        """
        timestamp = int(time.time() * 1000)  # Unique timestamp for layer names
        
        model = Sequential()
        
        # Input layer
        model.add(Dense(
            self.config["hidden_layers"][0],
            activation=self.config["activation"],
            input_shape=(input_dim,),
            name=f"input_layer_{timestamp}"
        ))
        model.add(BatchNormalization(name=f"bn_input_{timestamp}"))
        model.add(Dropout(self.config["dropout_rate"], name=f"dropout_input_{timestamp}"))
        
        # Hidden layers
        for i, units in enumerate(self.config["hidden_layers"][1:], 1):
            model.add(Dense(
                units,
                activation=self.config["activation"],
                name=f"hidden_layer_{i}_{timestamp}"
            ))
            model.add(BatchNormalization(name=f"bn_hidden_{i}_{timestamp}"))
            model.add(Dropout(self.config["dropout_rate"], name=f"dropout_hidden_{i}_{timestamp}"))
        
        # Output layer
        model.add(Dense(
            1,
            activation=self.config["output_activation"],
            name=f"output_layer_{timestamp}"
        ))
        
        # Compile model
        optimizer = Adam(learning_rate=self.config["learning_rate"])
        model.compile(
            optimizer=optimizer,
            loss=self.config["loss"],
            metrics=self.config["metrics"]
        )
        
        logger.info(f"Neural network built with {model.count_params()} parameters")
        return model
    
    def _get_callbacks(self) -> list:
        """
        Get training callbacks.
        
        Returns:
            List of Keras callbacks
        """
        callbacks = []
        
        # Early stopping
        if "early_stopping" in self.training_config["callbacks"]:
            early_stopping_config = self.training_config["early_stopping"]
            callbacks.append(EarlyStopping(**early_stopping_config))
        
        # Learning rate reduction
        if "reduce_lr" in self.training_config["callbacks"]:
            reduce_lr_config = self.training_config["reduce_lr"]
            callbacks.append(ReduceLROnPlateau(**reduce_lr_config))
        
        return callbacks
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
            
        Returns:
            Training history and metrics
        """
        logger.info("Starting neural network training")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val) if X_val is not None else None
        
        # Build model
        self.model = None  # Clear any existing model
        self.model = self._build_model(X_train.shape[1])
        
        # Get callbacks
        callbacks = self._get_callbacks()
        
        # Prepare validation data
        validation_data = (X_val_scaled, y_val) if X_val is not None else None
        
        # Train model
        self.history = self.model.fit(
            X_train_scaled, y_train,
            validation_data=validation_data,
            epochs=self.training_config["epochs"],
            batch_size=self.training_config["batch_size"],
            callbacks=callbacks,
            verbose=1
        )
        
        self.is_fitted = True
        
        # Calculate training metrics
        train_loss, train_accuracy, train_precision, train_recall = self.model.evaluate(
            X_train_scaled, y_train, verbose=0
        )
        
        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1_score": 2 * (train_precision * train_recall) / (train_precision + train_recall + 1e-8)
        }
        
        # Calculate validation metrics if validation data provided
        if X_val is not None:
            val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(
                X_val_scaled, y_val, verbose=0
            )
            metrics.update({
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1_score": 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
            })
        
        logger.info("Training completed!")
        logger.info(f"Final training accuracy: {train_accuracy:.4f}")
        if X_val is not None:
            logger.info(f"Final validation accuracy: {val_accuracy:.4f}")
        
        return metrics
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            threshold: Classification threshold
            
        Returns:
            Binary predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        y_pred_proba = self.model.predict(X_scaled, verbose=0)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        
        return y_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get prediction probabilities.
        
        Args:
            X: Features to predict on
            
        Returns:
            Prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        X_scaled = self.scaler.transform(X)
        y_pred_proba = self.model.predict(X_scaled, verbose=0)
        
        return y_pred_proba
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, 
                threshold: float = 0.5) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test targets
            threshold: Classification threshold
            
        Returns:
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
        
        # Calculate metrics
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            self.scaler.transform(X_test), y_test, verbose=0
        )
        
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-8)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        metrics = {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1_score": f1_score,
            "roc_auc": roc_auc,
            "loss": test_loss
        }
        
        logger.info("Evaluation completed!")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Test Precision: {test_precision:.4f}")
        logger.info(f"Test Recall: {test_recall:.4f}")
        logger.info(f"Test F1-Score: {f1_score:.4f}")
        logger.info(f"Test ROC-AUC: {roc_auc:.4f}")
        
        return metrics
    
    def get_feature_importance(self, X: np.ndarray, feature_names: list) -> Dict[str, float]:
        """
        Get feature importance using permutation importance.
        
        Args:
            X: Features
            feature_names: Names of features
            
        Returns:
            Dictionary of feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        # Get baseline score
        baseline_score = self.model.evaluate(
            self.scaler.transform(X), np.zeros(len(X)), verbose=0
        )[1]  # accuracy
        
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            # Create permuted data
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            # Get score with permuted feature
            permuted_score = self.model.evaluate(
                self.scaler.transform(X_permuted), np.zeros(len(X)), verbose=0
            )[1]
            
            # Calculate importance as difference from baseline
            importance_scores[feature_name] = baseline_score - permuted_score
        
        return importance_scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model.
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model.
        
        Args:
            filepath: Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.is_fitted = True
        logger.info(f"Model loaded from {filepath}")
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            String representation of model architecture
        """
        if self.model is None:
            return "Model not built yet"
        
        return self.model.summary()
