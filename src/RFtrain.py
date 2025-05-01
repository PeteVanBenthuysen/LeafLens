import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from pathlib import Path
import json
import config
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import joblib

# Set the path to RF splits directory
config.RF_SPLITS_DIR = Path("outputs/splits_rf") 

# Load training data
X_train = np.load(config.RF_SPLITS_DIR / "X_train_resampled.npy")
y_train = np.load(config.RF_SPLITS_DIR / "y_train_resampled.npy")

# Load validation data
X_val = np.load(config.RF_SPLITS_DIR / "X_val.npy")
y_val = np.load(config.RF_SPLITS_DIR / "y_val.npy")

# Load test data
X_test = np.load(config.RF_SPLITS_DIR / "X_test.npy")
y_test = np.load(config.RF_SPLITS_DIR / "y_test.npy")

print("Data loaded successfully.")
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [2, 4, 6, 8, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt'],
    'class_weight': ['balanced'],
    'bootstrap': [True]  
}

# Initialize base Random Forest 
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Setup StratifiedKFold for cross-validation
stratified_cv = StratifiedKFold(
    n_splits=5, 
    shuffle=True, 
    random_state=42
)

# Set up GridSearchCV
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=stratified_cv,
    scoring='accuracy',         
    verbose=1,                  # Basic output during fitting
    n_jobs=-1                   # Use all CPUs
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Print best results
print("\nBest Hyperparameters:")
print(grid_search.best_params_)
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Get the best estimator
best_rf_model = grid_search.best_estimator_

# Save the best RF model for later testing
model_save_path = config.RF_RESULTS_DIR / "best_rf_model.joblib"
joblib.dump(best_rf_model, model_save_path)
print(f"Best RF model saved to {model_save_path}")

# Evaluate on validation set
y_val_pred = best_rf_model.predict(X_val)

# Evaluate accuracy and classification report on validation set
print("\nValidation Set Performance:")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print("Classification Report:\n", classification_report(y_val, y_val_pred))

# ROC-AUC Score (macro averaged) on validation set
try:
    roc_auc = roc_auc_score(y_val, best_rf_model.predict_proba(X_val), multi_class='ovr', average='macro')
    print(f"Macro ROC-AUC Score: {roc_auc:.4f}")
except AttributeError:
    print("ROC-AUC could not be calculated (model might not have predict_proba).")

# Prepare validation results dictionary
validation_results = {
    'accuracy': accuracy_score(y_val, y_val_pred),
    'classification_report': classification_report(y_val, y_val_pred, output_dict=True),  # Use output_dict=True for clean JSON
}

# Try adding ROC-AUC to validation results
try:
    validation_results['roc_auc_macro'] = roc_auc_score(y_val, best_rf_model.predict_proba(X_val), multi_class='ovr', average='macro')
except AttributeError:
    validation_results['roc_auc_macro'] = None

# Save validation results as JSON
with open(config.RF_RESULTS_DIR / 'validation_results.json', 'w') as f:
    json.dump(validation_results, f, indent=4)

print(f"Validation results saved to {config.RF_RESULTS_DIR / 'validation_results.json'}")

# Generate and save confusion matrix
confusion_matrix_path = config.RF_RESULTS_DIR / 'confusion_matrix.png'
plot_rf_confusion_matrix_train(y_val, y_val_pred, confusion_matrix_path)

# Extract and pivot grid search results
scores_matrix = extract_and_pivot_gridsearch_train(grid_search)

# Plot and save heatmap
heatmap_path = config.RF_RESULTS_DIR / "n_estimators_vs_max_depth_heatmap.png"
plot_rf_heatmap_train(scores_matrix, heatmap_path)

# Sort by score
sorted_indices = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1]  # descending order
sorted_scores = np.array(grid_search.cv_results_['mean_test_score'])[sorted_indices]
sorted_params = np.array(grid_search.cv_results_['params'])[sorted_indices]

# Plot top 20 models
top20_plot_path = config.RF_RESULTS_DIR / 'top20_models.png'
plot_top20_gridsearch_models(sorted_scores, top20_plot_path, top_k=20)

# Prepare metadata to save
# Usage example in your RFtrain.py:
save_rf_gridsearch_metadata_train(
    grid_search=grid_search,
    param_grid=param_grid,
    confusion_matrix_path=confusion_matrix_path,
    top20_plot_path=top20_plot_path,
    heatmap_path=heatmap_path,
    save_dir=config.RF_RESULTS_DIR
)