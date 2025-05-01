import numpy as np
import json
import config
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from functions import plot_confusion_matrix_cnn_test, plot_prediction_explanation_grid, plot_cnn_probabilities_heatmap_test 

config.CNN_TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Load test data
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    config.CNN_TEST_DIR,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Load the trained model
model_path = config.CNN_RESULTS_DIR / "best_model.keras"
model = load_model(model_path)

# Predict
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = test_generator.classes
class_names = list(test_generator.class_indices.keys())

# Accuracy
acc = accuracy_score(y_true_labels, y_pred_labels)

# Classification report
cls_report = classification_report(y_true_labels, y_pred_labels, target_names=class_names, output_dict=True)

# ROC-AUC (macro)
try:
    roc_auc = roc_auc_score(y_true_labels, y_pred_probs, multi_class='ovr', average='macro')
except Exception:
    roc_auc = None

# Save results as JSON
test_results = {
    "accuracy": acc,
    "classification_report": cls_report,
    "roc_auc_macro": roc_auc
}
with open(config.CNN_TEST_RESULTS_DIR / "test_results.json", "w") as f:
    json.dump(test_results, f, indent=4)

print(f"Test results saved to {config.CNN_TEST_RESULTS_DIR / 'test_results.json'}")

cm_path = config.CNN_TEST_RESULTS_DIR / "confusion_matrix.png"
plot_confusion_matrix_cnn_test(y_true_labels, y_pred_labels, class_names, cm_path)
plt.show()

explanation_grid_path = config.CNN_TEST_RESULTS_DIR / "prediction_explanation_grid.png"
plot_prediction_explanation_grid(
    test_generator, y_pred_probs, y_pred_labels, class_names, explanation_grid_path, n_images=10
)

proba_path = config.CNN_TEST_RESULTS_DIR / "probabilities_heatmap.png"
plot_cnn_probabilities_heatmap_test(y_pred_probs, proba_path)
plt.show()

# Print weighted average metrics
weighted_avg = cls_report.get("weighted avg", {})
print("\nWeighted Average Metrics (all classes):")
for metric, value in weighted_avg.items():
    print(f"{metric}: {value:.4f}")

# Print ROC-AUC score
print(f"\nROC-AUC (macro): {roc_auc:.4f}" if roc_auc is not None else "\nROC-AUC (macro): N/A")