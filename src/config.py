from pathlib import Path
import os

# Project root
project_root = Path(__file__).resolve().parent.parent

# Base data and outputs directories
DATA_DIR = Path(os.getenv('DATA_DIR', project_root / 'data'))  # Use environment variable or default
OUTPUTS_DIR = Path(os.getenv('OUTPUTS_DIR', project_root / 'outputs'))  # Use environment variable or default
MASKED_DATA_DIR = Path(os.getenv('MASKED_DATA_DIR', OUTPUTS_DIR / 'masked_data'))
PLANT_VILLAGE_PATH = DATA_DIR / "PlantVillage"

# EDA directories
EDA_DIR = Path(os.getenv('EDA_DIR', OUTPUTS_DIR / 'EDA'))

# Masked data directories
PROCESSED_DIR = OUTPUTS_DIR / 'processed'
MASKED_ARRAYS_DIR = PROCESSED_DIR / 'masked_arrays'

# Splits directories
RF_SPLITS_DIR = Path(os.getenv('RF_SPLITS_DIR', OUTPUTS_DIR / 'splits_rf'))  # Use environment variable or default
CNN_SPLITS_DIR = Path(os.getenv('CNN_SPLITS_DIR', OUTPUTS_DIR / 'splits_cnn_transformed'))  # Augmented training data
CNN_VAL_DIR = Path(os.getenv('CNN_VAL_DIR', OUTPUTS_DIR / 'splits_cnn' / 'val'))  # Validation data (unaltered)
CNN_TEST_DIR = Path(os.getenv('CNN_TEST_DIR', OUTPUTS_DIR / 'splits_cnn' / 'test'))  # Test data (unaltered)

#RF split file paths
original_X_train_path = RF_SPLITS_DIR / 'X_train.npy'
original_y_train_path = RF_SPLITS_DIR / 'y_train.npy'
resampled_X_train_path = RF_SPLITS_DIR / 'X_train_resampled.npy'
resampled_y_train_path = RF_SPLITS_DIR / 'y_train_resampled.npy'
metadata_path = RF_SPLITS_DIR / 'resampling_metadata.json'

# Training results directories
RF_RESULTS_DIR = Path(os.getenv('RF_RESULTS_DIR', OUTPUTS_DIR / 'training_results' / 'rf'))
CNN_RESULTS_DIR = Path(os.getenv('CNN_RESULTS_DIR', OUTPUTS_DIR / 'training_results' / 'cnn'))

# Testing results directories
RF_TEST_RESULTS_DIR = Path(os.getenv('RF_TEST_RESULTS_DIR', OUTPUTS_DIR / 'testing_results' / 'rf'))
CNN_TEST_RESULTS_DIR = Path(os.getenv('CNN_TEST_RESULTS_DIR', OUTPUTS_DIR / 'testing_results' / 'cnn'))

# Ensure the necessary directories exist
MASKED_ARRAYS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
MASKED_DATA_DIR.mkdir(parents=True, exist_ok=True)
RF_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
EDA_DIR.mkdir(parents=True, exist_ok=True)
CNN_SPLITS_DIR.mkdir(parents=True, exist_ok=True)
CNN_VAL_DIR.mkdir(parents=True, exist_ok=True)
CNN_TEST_DIR.mkdir(parents=True, exist_ok=True)
RF_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CNN_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RF_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Debugging output
print(f"[CONFIG] DATA_DIR set to: {DATA_DIR}")
print(f"[CONFIG] MASKED_DATA_DIR set to: {MASKED_DATA_DIR}")
print(f"[CONFIG] OUTPUTS_DIR set to: {OUTPUTS_DIR}")
print(f"[CONFIG] RF_SPLITS_DIR set to: {RF_SPLITS_DIR}")
print(f"[CONFIG] CNN_SPLITS_DIR set to: {CNN_SPLITS_DIR}")
print(f"[CONFIG] CNN_VAL_DIR set to: {CNN_VAL_DIR}")
print(f"[CONFIG] CNN_TEST_DIR set to: {CNN_TEST_DIR}")
print(f"[CONFIG] RF_TEST_RESULTS_DIR set to: {RF_TEST_RESULTS_DIR}")