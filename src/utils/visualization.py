"""
Visualization utilities for Smart Grid Fault Detection System.

This module provides comprehensive visualization capabilities for model evaluation,
training monitoring, and data analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path

from ..config import REPORTS_DIR

logger = logging.getLogger(__name__)

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class VisualizationManager:
    """
    Manages all visualization tasks for the fault detection system.
    """
    
    def __init__(self, save_plots: bool = True, show_plots: bool = True):
        """
        Initialize visualization manager.
        
        Args:
            save_plots: Whether to save plots to files
            show_plots: Whether to display plots
        """
        self.save_plots = save_plots
        self.show_plots = show_plots
        self.reports_dir = REPORTS_DIR
        
        # Create reports directory if it doesn't exist
        self.reports_dir.mkdir(exist_ok=True)
        
        logger.info("VisualizationManager initialized")
    
    def plot_training_history(self, history: Any, model_name: str = "Neural Network") -> None:
        """
        Plot training history for neural network.
        
        Args:
            history: Keras training history object
            model_name: Name of the model for plot title
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Training Loss', linewidth=2)
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Training Precision', linewidth=2)
        axes[1, 0].plot(history.history['val_precision'], label='Validation Precision', linewidth=2)
        axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Training Recall', linewidth=2)
        axes[1, 1].plot(history.history['val_recall'], label='Validation Recall', linewidth=2)
        axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.reports_dir / f"{model_name.lower().replace(' ', '_')}_training_history.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            model_name: str = "Model") -> None:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model for plot title
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Fault', 'Fault'],
                   yticklabels=['No Fault', 'Fault'],
                   cbar_kws={'label': 'Count'})
        plt.title(f'{model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        # Add percentage annotations
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                percentage = (cm[i, j] / total) * 100
                plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                        ha='center', va='center', fontsize=10, color='red')
        
        if self.save_plots:
            filepath = self.reports_dir / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      model_name: str = "Model") -> None:
        """
        Plot ROC curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model for plot title
        """
        from sklearn.metrics import roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'{model_name} - ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            filepath = self.reports_dir / f"{model_name.lower().replace(' ', '_')}_roc_curve.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curve plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray,
                                   model_name: str = "Model") -> None:
        """
        Plot precision-recall curve.
        
        Args:
            y_true: True labels
            y_pred_proba: Prediction probabilities
            model_name: Name of the model for plot title
        """
        from sklearn.metrics import average_precision_score
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkorange', lw=2,
                label=f'PR curve (AP = {avg_precision:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'{model_name} - Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        if self.save_plots:
            filepath = self.reports_dir / f"{model_name.lower().replace(' ', '_')}_pr_curve.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curve plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_feature_importance(self, importance_scores: Dict[str, float], 
                              model_name: str = "Model", top_n: int = 15) -> None:
        """
        Plot feature importance.
        
        Args:
            importance_scores: Dictionary of feature importance scores
            model_name: Name of the model for plot title
            top_n: Number of top features to display
        """
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:top_n]
        
        features, scores = zip(*top_features)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(len(features)), scores, color='skyblue', alpha=0.8)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title(f'{model_name} - Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{score:.3f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.reports_dir / f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_data_distribution(self, data: pd.DataFrame, target_column: str) -> None:
        """
        Plot data distribution and target variable analysis.
        
        Args:
            data: DataFrame with data
            target_column: Name of target column
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Target distribution
        target_counts = data[target_column].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['No Fault', 'Fault'], 
                      autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('Target Variable Distribution', fontsize=14, fontweight='bold')
        
        # Numerical features distribution
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            sample_col = numerical_cols[0]
            axes[0, 1].hist(data[sample_col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 1].set_title(f'Distribution of {sample_col}', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel(sample_col)
            axes[0, 1].set_ylabel('Frequency')
        
        # Correlation heatmap (sample of features)
        if len(numerical_cols) > 1:
            sample_cols = numerical_cols[:10]  # Limit to 10 features for readability
            corr_matrix = data[sample_cols].corr()
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                       ax=axes[1, 0], cbar_kws={'shrink': 0.8})
            axes[1, 0].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Missing values
        missing_data = data.isnull().sum()
        missing_data = missing_data[missing_data > 0]
        if len(missing_data) > 0:
            axes[1, 1].bar(range(len(missing_data)), missing_data.values, color='orange', alpha=0.7)
            axes[1, 1].set_xticks(range(len(missing_data)))
            axes[1, 1].set_xticklabels(missing_data.index, rotation=45, ha='right')
            axes[1, 1].set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
            axes[1, 1].set_ylabel('Number of Missing Values')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                           transform=axes[1, 1].transAxes, fontsize=14)
            axes[1, 1].set_title('Missing Values Analysis', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.reports_dir / "data_distribution_analysis.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Data distribution plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        Plot model comparison metrics.
        
        Args:
            comparison_df: DataFrame with model comparison metrics
        """
        if comparison_df.empty:
            logger.warning("No model comparison data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for i, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                ax = axes[i//2, i%2]
                bars = ax.bar(comparison_df['Model'], comparison_df[metric], 
                            color='lightblue', alpha=0.8, edgecolor='black')
                ax.set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
                ax.set_ylabel(metric)
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, comparison_df[metric]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=10)
                
                # Rotate x-axis labels if needed
                if len(comparison_df['Model'].iloc[0]) > 10:
                    ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.save_plots:
            filepath = self.reports_dir / "model_comparison.png"
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {filepath}")
        
        if self.show_plots:
            plt.show()
        else:
            plt.close()
    
    def create_summary_report(self, results: Dict[str, Any], 
                            model_name: str = "Neural Network") -> None:
        """
        Create a comprehensive summary report with all visualizations.
        
        Args:
            results: Model training results
            model_name: Name of the model
        """
        logger.info(f"Creating summary report for {model_name}")
        
        # Plot training history if available
        if 'history' in results:
            self.plot_training_history(results['history'], model_name)
        
        # Plot confusion matrix
        if 'test_data' in results and 'predictions' in results:
            X_test, y_test = results['test_data']
            y_pred = results['predictions']
            self.plot_confusion_matrix(y_test, y_pred, model_name)
        
        # Plot ROC curve
        if 'test_data' in results and 'probabilities' in results:
            X_test, y_test = results['test_data']
            y_pred_proba = results['probabilities']
            self.plot_roc_curve(y_test, y_pred_proba, model_name)
            self.plot_precision_recall_curve(y_test, y_pred_proba, model_name)
        
        logger.info(f"Summary report created for {model_name}")
