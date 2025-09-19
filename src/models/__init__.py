"""
Models module for Smart Grid Fault Detection System.

This module contains machine learning models for fault detection.
"""

from .neural_network import NeuralNetwork
from .model_trainer import ModelTrainer

__all__ = [
    'NeuralNetwork',
    'ModelTrainer',
]
