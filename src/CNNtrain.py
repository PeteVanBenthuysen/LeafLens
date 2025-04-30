# Path management to ensure local imports work
import sys
from pathlib import Path

# Basic libraries
import os
import warnings
warnings.filterwarnings("ignore")
import json 
import numpy as np
import io

# Deep learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Input, layers, Model

# Evaluation utilities
from sklearn.metrics import classification_report

# Project-specific modules
import config
from functions import (
    evaluate_and_visualize_model,
    build_improved_cnn_model,
    tta_predict,
    save_cnn_metrics_to_json,
    save_model_summary_arch_weights
)

# Load data
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    config.CNN_SPLITS_DIR / "train",
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    config.CNN_VAL_DIR,
    target_size=(128, 128),
    batch_size=64,
    class_mode='categorical'
)

input_shape = (128, 128, 3)
num_classes = len(train_generator.class_indices)
model = build_improved_cnn_model(input_shape, num_classes)
model.summary()

# Callbacks
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1)
lr_schedule = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
checkpoint_path = config.CNN_RESULTS_DIR / "best_model.keras"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)

print("Train class distribution:", np.bincount(train_generator.classes))
print("Val class distribution:", np.bincount(val_generator.classes))
print("Class indices:", train_generator.class_indices)

# Train the model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=50,
    callbacks=[early_stop, lr_schedule, model_checkpoint]
)

if hasattr(model, "to_json") and model.__class__.__name__ == "Sequential":
    model_config = model.get_config()
    input_shape = model.input_shape[1:]
    inputs = Input(shape=input_shape, name="gradcam_input")
    x = inputs
    for layer_conf in model_config['layers'][1:]:
        layer = layers.deserialize(layer_conf)
        x = layer(x)
    functional_model = Model(inputs, x)
    functional_model.set_weights(model.get_weights())
else:
    functional_model = model

# Evaluate and visualize results
class_names = list(train_generator.class_indices.keys())
evaluate_and_visualize_model(
    model=model,
    functional_model=functional_model,
    val_generator=val_generator,
    history=history,
    class_names=class_names
)

# Predict on validation set without TTA and save classification report
val_generator.reset()
y_pred_probs = model.predict(val_generator, verbose=1)
y_pred_labels = np.argmax(y_pred_probs, axis=1)
y_true_labels = val_generator.classes

# Classification report
non_tta_report = classification_report(
    y_true_labels, y_pred_labels,
    target_names=class_names,
    output_dict=True
)

# Save report as JSON
non_tta_report_path = config.CNN_RESULTS_DIR / "non_tta_classification_report.json"
with open(non_tta_report_path, 'w') as f:
    json.dump(non_tta_report, f, indent=4)

print(f"Non-TTA classification report saved to: {non_tta_report_path}")

# Save model summary, architecture, and weights
save_model_summary_arch_weights(
    model,
    output_path=config.CNN_RESULTS_DIR / "model_summary_arch_weights.json"
)

# Run TTA
print("\nRunning TTA on validation set...")
y_pred_tta, y_true_tta = tta_predict(model, val_generator, num_classes, tta_steps=5)
y_pred_tta_labels = np.argmax(y_pred_tta, axis=1)
y_true_tta_labels = np.argmax(y_true_tta, axis=1)

print("\nTTA Classification Report:\n")
tta_report = classification_report(y_true_tta_labels, y_pred_tta_labels, target_names=class_names, output_dict=True)
print(classification_report(y_true_tta_labels, y_pred_tta_labels, target_names=class_names))

# Variable for Validation ROC-AUC score
y_pred_tta_probs = model.predict(val_generator)
y_pred_tta_labels = np.argmax(y_pred_tta_probs, axis=1)

# Save metrics and TTA results to JSON
save_cnn_metrics_to_json(
    output_dir=config.CNN_RESULTS_DIR,
    history=history,
    class_names=class_names,
    y_true=y_true_tta_labels,
    y_pred=y_pred_tta_labels,
    y_pred_probs=y_pred_tta_probs,
    tta_report=tta_report
)
