import os
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Data files
RAW_DATA = DATA_DIR / "spam.csv"
PROCESSED_DATA = DATA_DIR / "processed_data.pkl"

# Model files
MODEL_FILE = MODELS_DIR / "spam_classifier.pkl"

# Model parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42