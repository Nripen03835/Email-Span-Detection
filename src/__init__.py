"""
Spam Detection Module

This package provides functionality for detecting spam messages using machine learning.
It includes data preprocessing, model training, and prediction capabilities.

Modules:
- data_preprocessing: Handles text cleaning and feature extraction
- model: Contains the SpamClassifier class
- train: Script for training and saving the model
- predict: Script for making predictions with the trained model
"""

from .data_preprocessing import clean_text, load_and_preprocess_data, get_train_test_data
from .model import SpamClassifier
from .predict import SpamPredictor

__all__ = [
    'clean_text',
    'load_and_preprocess_data',
    'get_train_test_data',
    'SpamClassifier',
    'SpamPredictor'
]

__version__ = '1.0.0'