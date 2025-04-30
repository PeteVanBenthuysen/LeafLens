# Basic libraries
import os
import random
import io
import collections
from pathlib import Path
import json
import numpy as np
import pandas as pd

# Image handling
from PIL import Image
import cv2

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Progress bar
from tqdm import tqdm

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model, Sequential, load_model, clone_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.regularizers import l2

# Sklearn metrics
from sklearn.metrics import confusion_matrix, classification_report

# Project config
import config

# Function to plot bar chart for a specific crop
def plot_crop_distribution(crop_name, crop_data):
    labels = ["Healthy"] + list(crop_data["infected"].keys())
    counts = [crop_data["healthy"]] + list(crop_data["infected"].values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=["lightgreen"] + ["salmon"] * len(crop_data["infected"]))
    plt.xticks(rotation=90)
    plt.title(f"{crop_name} - Healthy vs. Infected Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.show()

# Function to apply a leaf mask to a single image
def apply_leaf_mask(image: np.ndarray) -> np.ndarray:
    """
    Applies a simple green color threshold to create a leaf mask.
    Modify this function if you have a better masking method.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Define green color range in HSV
    lower_green = np.array([15, 20, 20])    # softer hue, softer saturation, softer brightness
    upper_green = np.array([110, 255, 255]) # still allows yellowish and slightly brownish tones

    # Threshold to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Clean noise in the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Apply mask to original image
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

# Function to apply masking to all images in the data directory
def mask_and_save_all_images():
    print("[INFO] Starting leaf masking process...")

    # Make sure target dir exists
    config.MASKED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Loop through all images
    image_paths = list(config.DATA_DIR.glob('**/*.jpg')) + list(config.DATA_DIR.glob('**/*.png'))

    print(f"[INFO] Found {len(image_paths)} images in DATA_DIR.")

    for img_path in image_paths:
        # Open image
        image = Image.open(img_path).convert('RGB')
        image_np = np.array(image)

        # Apply mask
        masked_np = apply_leaf_mask(image_np)

        # Convert back to PIL Image
        masked_img = Image.fromarray(masked_np)

        # Save to mirrored path in MASKED_DATA_DIR
        relative_path = img_path.relative_to(config.DATA_DIR)  # Keeps same folder structure
        save_path = config.MASKED_DATA_DIR / relative_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        masked_img.save(save_path, quality=95, optimize=True)

    print(f"[INFO] Masking complete. Masked images saved to: {config.MASKED_DATA_DIR}")

if __name__ == "__main__":
    mask_and_save_all_images()

# Function to check if a split directory is complete
def is_split_complete(split_dir):
    """Check if a split directory (train/val/test) is complete."""
    if not split_dir.exists():
        return False  # Directory doesn't exist
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir() and not any(class_dir.iterdir()):  # Check if class folder is empty
            return False  # Empty class folder found
    return True  # All class folders are populated

# Function to save split data
def save_cnn_split(split_dir, data, labels, split_name):
    """
    Saves images and labels into split directories (train/val/test) organized by class.
    """
    split_subdir = split_dir / split_name
    split_subdir.mkdir(parents=True, exist_ok=True)
    
    for idx, (img, label) in enumerate(zip(data, labels)):
        category_dir = split_subdir / f"class_{label}"  # Save by class label
        category_dir.mkdir(parents=True, exist_ok=True)
        img_path = category_dir / f"img_{idx}.png"
        Image.fromarray((img * 255).astype(np.uint8)).save(img_path)

# Function to show random masked images
def show_random_images(images, labels, class_names, n=5):
    """
    Display n random images with their labels.
    """
    plt.figure(figsize=(15, 5))
    for i in range(n):
        idx = random.randint(0, len(images) - 1)
        img = images[idx]
        label = labels[idx]

        plt.subplot(1, n, i + 1)
        plt.imshow(img.astype('float32') / 255.0)
        plt.title(class_names[label])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Autoencoder graph function 
def plot_compressed_scatter(features_2d, labels, title, save_path):
    """
    Plot 2D compressed features colored by labels or clusters.
    
    Args:
        features_2d (np.ndarray): 2D features (shape: n_samples x 2)
        labels (np.ndarray): Labels or cluster IDs for coloring
        title (str): Title for the plot
        save_path (Path): Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', alpha=0.7, s=15)
    plt.colorbar(scatter, label='Label')
    plt.title(title)
    plt.xlabel('Compressed Feature 1')
    plt.ylabel('Compressed Feature 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_cluster_scatter(features_2d, cluster_labels, title, save_path):
    """
    Plot a scatter plot of clustered features (e.g., from PCA or Autoencoder compression).

    Args:
        features_2d (np.ndarray): 2D reduced feature array (n_samples, 2)
        cluster_labels (np.ndarray): Cluster assignments (e.g., from KMeans)
        title (str): Plot title
        save_path (Path): Where to save the plot
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=cluster_labels, cmap='tab20', alpha=0.7, s=15)
    plt.colorbar(scatter, label='KMeans Cluster')
    plt.title(title)
    plt.xlabel('Compressed Feature 1')
    plt.ylabel('Compressed Feature 2')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to plot autoencoder loss curve
def plot_autoencoder_loss_curve(history, save_path):
    """
    Plot and save the autoencoder training and validation loss curves.

    Args:
        history: Keras History object containing 'loss' and 'val_loss'.
        save_path (Path): Path to save the loss curve plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Autoencoder Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to build Autoencoder
def build_autoencoder(input_dim, bottleneck_dim):
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(input_layer)
    x = layers.Dense(256, activation='relu')(x)
    bottleneck = layers.Dense(bottleneck_dim, activation='relu', name='bottleneck')(x)
    x = layers.Dropout(0.3)(bottleneck)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(512, activation='relu')(x)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(x)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

# Function to plot bottleneck dimension vs final validation loss
def plot_bottleneck_vs_loss(bottleneck_results, save_path):
    """
    Plot and save bottleneck dimension vs final validation loss.

    Args:
        bottleneck_results (dict): Dictionary mapping bottleneck dimensions to validation loss.
        save_path (Path): Where to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(list(bottleneck_results.keys()), list(bottleneck_results.values()), marker='o')
    plt.title('Bottleneck Size vs Final Validation Loss')
    plt.xlabel('Bottleneck Dimension')
    plt.ylabel('Final Validation MSE Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved bottleneck vs loss graph to {save_path}")

# Function to plot final autoencoder loss
def plot_final_autoencoder_loss(history, best_bottleneck_dim, save_path):
    """
    Plot and save the final optimized autoencoder training and validation loss curves.

    Args:
        history: Keras History object containing 'loss' and 'val_loss'.
        best_bottleneck_dim (int): The best bottleneck dimension selected.
        save_path (Path): Path to save the generated plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title(f'Final Autoencoder Loss Curve (Optimized Bottleneck {best_bottleneck_dim})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved final optimized loss curve to {save_path}")

# Function to evaluate and visualize the CNN model
def evaluate_and_visualize_model(model, functional_model, val_generator, history, class_names):
    print("Starting evaluation and visualization...")
    output_dir = Path(config.CNN_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    # Save training curves
    print("Plotting training curves...")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Accuracy Over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curve.png')
    plt.close()
    print("Training curves saved.")

    # Confusion matrix
    print("Generating confusion matrix...")
    val_generator.reset()
    try:
        y_true = val_generator.classes
        y_pred_probs = functional_model.predict(val_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        print("Predictions completed.")
    except Exception as e:
        print(f"Failed during prediction: {e}")
        return

    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print("Confusion matrix generated.")

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    plt.close()
    print("Confusion matrix saved.")

    # Prediction visualizations
    print("Generating prediction visualizations...")
    val_generator.reset()
    try:
        x_val, y_val = next(val_generator)
        indices = np.random.choice(len(x_val), size=8, replace=False)
    except Exception as e:
        print(f"Failed to fetch validation data: {e}")
        return

    plt.figure(figsize=(16, 8))
    for i, idx in enumerate(indices):
        img = x_val[idx]
        true_label = np.argmax(y_val[idx])
        pred_probs = functional_model.predict(img[np.newaxis, ...])[0]
        pred_label = np.argmax(pred_probs)

        plt.subplot(4, 4, i * 2 + 1)
        if img.dtype == np.float32 and img.max() <= 1.0:
            plt.imshow(img)
        else:
            plt.imshow(img.astype('float32') / 255.0)
        plt.axis('off')
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f"{class_names[pred_label]} {pred_probs[pred_label]*100:.1f}% ({class_names[true_label]})", color=color)

        plt.subplot(4, 4, i * 2 + 2)
        bars = plt.bar(range(len(class_names)), pred_probs, color='gray')
        bars[pred_label].set_color('green')
        plt.xticks(range(len(class_names)), class_names, rotation=45)
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_grid.png')
    plt.close()
    print("Prediction visualizations saved.")

    # Grad-CAM visualizations
    print("Generating Grad-CAM visualizations...")
    gradcam_dir = output_dir / 'gradcam'
    gradcam_dir.mkdir(exist_ok=True)

    # Find all Conv2D layers
    conv_layers = [layer for layer in functional_model.layers if isinstance(layer, Conv2D)]
    if len(conv_layers) < 2:
        return

    # Target the last Conv2D layer for Grad-CAM
    target_layer = conv_layers[-3] # Adjust index as needed
    print(f"Using Conv2D layer for Grad-CAM: {target_layer.name}")

    # Use this layer for Grad-CAM
    grad_model = Model(
        inputs=functional_model.input,
        outputs=[functional_model.get_layer(target_layer.name).output, functional_model.output]
    )

    def make_gradcam_heatmap(img_array, pred_index=None):
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_tensor, training=False)
            if pred_index is None:
                pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]
        grads = tape.gradient(class_channel, conv_outputs)
        if grads is None:
            print("Gradients are None. Skipping this image.")
            return np.zeros(conv_outputs.shape[1:3])
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()

    print("Generating Grad-CAM heatmaps...")
    for i, idx in enumerate(indices):
        img = x_val[idx]
        img_array = np.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(img_array)

        # Normalize heatmap to [0, 1]
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        else:
            heatmap = np.zeros_like(heatmap)

        # Threshold to highlight strong activations
        threshold = 0.1  # Try lowering this if you see almost nothing
        heatmap[heatmap < threshold] = 0

        # Resize and colorize
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_TURBO)

        # Save the heatmap alone for debugging
        cv2.imwrite(str(gradcam_dir / f'heatmap_{i}.png'), heatmap_color)

        # Convert image to uint8 if needed
        if img.dtype == np.float32 or img.max() <= 1.0:
            original_img = np.uint8(img * 255)
        else:
            original_img = img

    print(f"[INFO] Grad-CAM visualizations saved to {gradcam_dir}")

# Function to build an improved CNN model with L2 + intermediate dropout
def build_improved_cnn_model(input_shape, num_classes, l2_value=0.0001):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (5, 5), activation='relu', kernel_regularizer=l2(l2_value)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        

        Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        

        Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(l2_value)),
        MaxPooling2D((2, 2)),
        BatchNormalization(),
        

        GlobalAveragePooling2D(),
        Dense(256, activation='relu', kernel_regularizer=l2(l2_value)),
        Dropout(0.5),  # Final strong dropout
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
    )

    return model

# Function to perform TTA (Test Time Augmentation) predictions
def tta_predict(model, generator, num_classes, tta_steps=5):
    predictions = []
    true_labels = []
    for i in range(len(generator)):
        x_batch, y_batch = generator[i]
        batch_preds = np.zeros((x_batch.shape[0], num_classes))
        for _ in range(tta_steps):
            augmented = x_batch.copy()
            if np.random.rand() > 0.5:
                augmented = augmented[:, :, ::-1, :]
            batch_preds += model.predict(augmented, verbose=0)
        predictions.extend(batch_preds / tta_steps)
        true_labels.extend(y_batch)
    return np.array(predictions), np.array(true_labels)

def save_cnn_metrics_to_json(output_dir, history, class_names, y_true, y_pred, y_pred_probs, tta_report=None):
    # Calculate accuracy
    acc = accuracy_score(y_true, y_pred)

    # One-hot encode true labels for ROC AUC
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    # Compute multiclass ROC AUC (One-vs-Rest)
    try:
        roc_auc = roc_auc_score(y_true_bin, y_pred_probs, multi_class='ovr')
    except ValueError:
        roc_auc = None  # In case there aren't enough classes for AUC

    # Assemble metrics
    result_dict = {
        "accuracy": acc,
        "roc_auc": roc_auc,
        "class_names": class_names,
        "training_history": {
            "loss": history.history.get("loss"),
            "val_loss": history.history.get("val_loss"),
            "accuracy": history.history.get("accuracy"),
            "val_accuracy": history.history.get("val_accuracy")
        },
        "tta_classification_report": tta_report
    }

    # Save to file
    with open(Path(output_dir) / "validation_results.json", "w") as f:
        json.dump(result_dict, f, indent=4)

# Function to save model summary, architecture, and weights
def save_model_summary_arch_weights(model, output_path):
    # Save model summary to string
    summary_stream = io.StringIO()
    model.summary(print_fn=lambda x: summary_stream.write(x + '\n'))
    summary_str = summary_stream.getvalue()

    # Save architecture
    model_json = model.to_json()

    # Save weights
    weights = model.get_weights()
    weights_serializable = [w.tolist() for w in weights]

    # Combine all into one dictionary
    model_metadata = {
        "summary": summary_str,
        "architecture_json": model_json,
        "weights": weights_serializable
    }

    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(model_metadata, f, indent=4)