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
cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save confusion matrix as PNG
confusion_matrix_path = config.RF_RESULTS_DIR / 'confusion_matrix.png'
plt.savefig(confusion_matrix_path)
plt.close()

# Print saved graph path
print(f"Confusion matrix saved to {confusion_matrix_path}")

# Extract GridSearchCV results and create heatmap for n_estimators vs max_depth
results_df = pd.DataFrame(grid_search.cv_results_)

# Filter for default values of other hyperparameters
filtered = results_df[
    (results_df.param_min_samples_split == 2) &
    (results_df.param_min_samples_leaf == 1) &
    (results_df.param_max_features == 'sqrt') &
    (results_df.param_class_weight == 'balanced') &
    (results_df.param_bootstrap == True)
]

# Pivot to get mean_test_score matrix
scores_matrix = filtered.pivot(index='param_n_estimators', columns='param_max_depth', values='mean_test_score')

# Plot and save heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(scores_matrix, annot=True, fmt=".4f", cmap='viridis')
plt.title('GridSearchCV Accuracy: n_estimators vs max_depth')
plt.xlabel('max_depth')
plt.ylabel('n_estimators')
plt.tight_layout()

heatmap_path = config.RF_RESULTS_DIR / "n_estimators_vs_max_depth_heatmap.png"
plt.savefig(heatmap_path)
plt.close()
print(f"Heatmap saved to {heatmap_path}")

# Sort by score
sorted_indices = np.argsort(grid_search.cv_results_['mean_test_score'])[::-1]  # descending order
sorted_scores = np.array(grid_search.cv_results_['mean_test_score'])[sorted_indices]
sorted_params = np.array(grid_search.cv_results_['params'])[sorted_indices]

# Plot top 20 models
top_k = 20
plt.figure(figsize=(14, 6))
plt.plot(range(top_k), sorted_scores[:top_k], marker='o')
plt.title('Top 20 GridSearchCV Results')
plt.xlabel('Model Rank')
plt.ylabel('Mean Cross-Validation Score')
plt.grid(True)
plt.xticks(range(top_k), [f"Model {i+1}" for i in range(top_k)], rotation=45)
plt.tight_layout()
plt.show()

# Save plot
top20_plot_path = config.RF_RESULTS_DIR / 'top20_models.png'
plt.savefig(top20_plot_path)
plt.close()

# Prepare metadata to save
results_metadata = {
    'best_params': grid_search.best_params_,
    'best_score': grid_search.best_score_,
    'param_grid': param_grid,
    'cv_results_summary': {
        'mean_test_score': list(grid_search.cv_results_['mean_test_score']),
        'params': [str(p) for p in grid_search.cv_results_['params']]
    },
    'saved_graphs': {
        'confusion_matrix': confusion_matrix_path.name,
        'top20_models_plot': top20_plot_path.name,
        'heatmap': heatmap_path.name
    }
}

# Save metadata JSON
with open(config.RF_RESULTS_DIR / 'gridsearch_metadata.json', 'w') as f:
    json.dump(results_metadata, f, indent=4)

print(f"\nGridSearch metadata saved to {config.RF_RESULTS_DIR / 'gridsearch_metadata.json'}")