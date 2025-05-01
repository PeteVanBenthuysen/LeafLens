import numpy as np
import joblib
import json
import config
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from functions import plot_rf_confusion_matrix_test, plot_rf_probabilities_heatmap_test, plot_rf_feature_importance_test

# Load test data
X_test = np.load(config.RF_SPLITS_DIR / "X_test.npy")
y_test = np.load(config.RF_SPLITS_DIR / "y_test.npy")

# Load the trained model
model_path = config.RF_RESULTS_DIR / "best_rf_model.joblib"
rf_model = joblib.load(model_path)

# Predict
y_pred = rf_model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

# Classification report
cls_report = classification_report(y_test, y_pred, output_dict=True)

# ROC-AUC (macro)
try:
    roc_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test), multi_class='ovr', average='macro')
except Exception:
    roc_auc = None

# Save results as JSON
test_results = {
    "accuracy": acc,
    "classification_report": cls_report,
    "roc_auc_macro": roc_auc
}
with open(config.RF_TEST_RESULTS_DIR / "test_results.json", "w") as f:
    json.dump(test_results, f, indent=4)

print(f"Test results saved to {config.RF_TEST_RESULTS_DIR / 'test_results.json'}")

# Feature importance plot
feature_importance_path = config.RF_TEST_RESULTS_DIR / "feature_importance.png"
plot_rf_feature_importance_test(rf_model, feature_names, feature_importance_path)

# Confusion matrix
cm_path = config.RF_TEST_RESULTS_DIR / "confusion_matrix.png"
plot_rf_confusion_matrix_test(y_test, y_pred, cm_path)
plt.show()
# Heatmap of probabilities
try:
    proba = rf_model.predict_proba(X_test)
    proba_path = config.RF_TEST_RESULTS_DIR / "probabilities_heatmap.png"
    plot_rf_probabilities_heatmap_test(proba, proba_path)
    plt.show()
except Exception as e:
    print(f"Could not plot probabilities heatmap: {e}")

with open(config.RF_TEST_RESULTS_DIR / "test_results.json", "r") as f:
    results = json.load(f)

# Extract weighted average metrics
weighted_avg = results["classification_report"]["weighted avg"]
print("\nWeighted Average Metrics (all classes):")
for metric, value in weighted_avg.items():
    print(f"{metric}: {value:.4f}")

# Extract and print ROC-AUC (macro)
roc_auc = results.get("roc_auc_macro", None)
print(f"\nROC-AUC (macro): {roc_auc:.4f}" if roc_auc is not None else "\nROC-AUC (macro): N/A")