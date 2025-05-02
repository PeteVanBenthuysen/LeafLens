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
import tensorflow_probability as tfp
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import Model, Sequential, load_model, clone_model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.regularizers import l2

# Torchvision
import torchvision.transforms as transforms

# Sklearn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# Project config
import config

# Data masking functions

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

# Function to load masked images and labels from a given path
def load_images_and_labels_from_path(dataset_path, categories, image_size=(128, 128)):
    """
    Loads images and labels from a dataset path (e.g., masked or unmasked).
    Args:
        dataset_path (Path): Path to the dataset folder (e.g., MASKED_DATA_DIR / "PlantVillage").
        categories (list): List of category names (subfolder names).
        image_size (tuple): Desired image size (width, height).
    Returns:
        images (np.ndarray): Array of loaded images.
        labels (np.ndarray): Array of integer labels.
    """
    images = []
    labels = []
    for label_idx, category in enumerate(categories):
        category_path = dataset_path / category
        for img_name in os.listdir(category_path):
            img_path = category_path / img_name
            if Path(img_path).is_file():
                img = Image.open(img_path).convert('RGB')
                img = img.resize(image_size)
                img_array = np.array(img)
                images.append(img_array)
                labels.append(label_idx)
    return np.array(images), np.array(labels)

# EDA functions

# Function to load images and labels from the PlantVillage dataset
def load_images_and_labels(plant_village_path, categories, image_size=(128, 128)):
    """
    Loads images and labels from the PlantVillage dataset.

    Args:
        plant_village_path (Path): Path to the PlantVillage folder.
        categories (list): List of category names (subfolder names).
        image_size (tuple): Desired image size (width, height).

    Returns:
        images (np.ndarray): Array of loaded images.
        labels (np.ndarray): Array of integer labels.
        skipped_images (dict): Dictionary of skipped image counts per category.
    """
    import os
    from PIL import Image
    import numpy as np

    images = []
    labels = []
    skipped_images = {}

    for label_idx, category in enumerate(categories):
        category_path = plant_village_path / category
        skipped_images[category] = 0
        for img_name in os.listdir(category_path):
            img_path = category_path / img_name
            if img_path.is_file():
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = img.resize(image_size)
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label_idx)
                except (IOError, OSError) as e:
                    print(f"Skipped invalid image {img_path} under category '{category}': {e}")
                    skipped_images[category] += 1

    images = np.array(images)
    labels = np.array(labels)
    return images, labels, skipped_images

# Function to plot original class distribution
def plot_class_distribution(labels, categories, title='Number of Images per Class', save_path=None):
    """
    Plots a bar chart of class distribution and optionally saves it.
    
    Args:
        labels (array-like): List or array of integer class labels.
        categories (list): List of class names (ordered by label index).
        title (str): Title for the plot.
        save_path (str or Path, optional): If provided, saves the plot to this path.
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    labels_df = pd.DataFrame({'label': labels})
    class_counts = labels_df['label'].value_counts().sort_index()

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(categories)), class_counts.values, tick_label=categories, color='skyblue')
    plt.xticks(rotation=90)
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.grid(axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# Function to group and report crop distribution
def group_and_report_crop_distribution(class_counts, categories, plot_func=None, save_dir=None):
    """
    Groups class counts into healthy and infected for each crop and optionally prints and plots the results.

    Args:
        class_counts (dict or pd.Series): Mapping from label index to count.
        categories (list): List of class names (ordered by label index).
        plot_func (callable, optional): Function to plot crop distribution. Should accept (crop_name, crop_data, save_path).
        save_dir (Path or str, optional): Directory to save plots. If None, plots are not saved.

    Returns:
        crops (dict): Nested dictionary with healthy and infected counts per crop.
    """
    crops = {
        "Potato": {"healthy": 0, "infected": {}},
        "Pepper": {"healthy": 0, "infected": {}},
        "Tomato": {"healthy": 0, "infected": {}}
    }

    for label_idx, count in class_counts.items():
        category = categories[label_idx]
        if "Potato" in category:
            if "healthy" in category.lower():
                crops["Potato"]["healthy"] += count
            else:
                crops["Potato"]["infected"][category] = count
        elif "Pepper" in category:
            if "healthy" in category.lower():
                crops["Pepper"]["healthy"] += count
            else:
                crops["Pepper"]["infected"][category] = count
        elif "Tomato" in category:
            if "healthy" in category.lower():
                crops["Tomato"]["healthy"] += count
            else:
                crops["Tomato"]["infected"][category] = count

    # Print and plot
    for crop_name, crop_data in crops.items():
        print(f"\n--- {crop_name} Dataset ---")
        print(f"Healthy: {crop_data['healthy']} images")
        for category, count in crop_data["infected"].items():
            print(f"Infected ({category}): {count} images")
        if plot_func:
            save_path = None
            if save_dir is not None:
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                save_path = Path(save_dir) / f"{crop_name.lower()}_distribution.png"
            plot_func(crop_name, crop_data, save_path=save_path)  # Show and save

    return crops

# Function to plot bar chart for a specific crop
def plot_crop_distribution(crop_name, crop_data, save_path=None):
    labels = ["Healthy"] + list(crop_data["infected"].keys())
    counts = [crop_data["healthy"]] + list(crop_data["infected"].values())

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts, color=["lightgreen"] + ["salmon"] * len(crop_data["infected"]))
    plt.xticks(rotation=90)
    plt.title(f"{crop_name} - Healthy vs. Infected Categories")
    plt.xlabel("Category")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.show()

# Data splitting functions

# Function to check if a folder contains the expected data
def is_folder_complete(folder_path, labels):
    folder_path = Path(folder_path)
    if not folder_path.exists():
        return False  # Folder doesn't exist, so it's not complete

    # Check if all class folders exist and contain files
    for label in set(labels):  # Unique class labels
        class_dir = folder_path / f"class_{label}"
        if not class_dir.exists() or not any(class_dir.iterdir()):  # Check if folder exists and is not empty
            return False  # Missing or empty class folder

    return True  # Folder is complete

# Helper function to check if a file exists and is not empty
def is_valid_file(path):
    return path.exists() and path.stat().st_size > 0

# Function to check if a split directory is complete
def is_split_complete(split_dir):
    """Check if a split directory (train/val/test) is complete."""
    if not split_dir.exists():
        return False  # Directory doesn't exist
    for class_dir in split_dir.iterdir():
        if class_dir.is_dir() and not any(class_dir.iterdir()):  # Check if class folder is empty
            return False  # Empty class folder found
    return True  # All class folders are populated

# Function to print class distribution with names
def print_class_distribution_with_names(counter, categories, title):
    print(title)
    for idx, count in sorted(counter.items()):
        class_name = categories[idx]
        print(f"{class_name}: {count}")
    print()

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

# Split dataset class imbalance functions

def count_images_per_class(directory, categories=None):
    directory = Path(directory)
    if not directory.exists():
        print(f"Directory does not exist: {directory}")
        return {}  # Return an empty dictionary if the directory is missing

    counts = {}
    if categories is not None:
        for idx, class_name in enumerate(categories):
            class_folder = f"class_{idx}"
            class_path = directory / class_folder
            if class_path.exists() and class_path.is_dir():
                counts[class_name] = len(list(class_path.iterdir()))
            else:
                counts[class_name] = 0
    else:
        for class_folder in os.listdir(directory):
            class_path = os.path.join(directory, class_folder)
            if os.path.isdir(class_path):
                counts[class_folder] = len(os.listdir(class_path))
    return counts

def plot_rf_class_distribution(labels, title, categories=None, save_path=None):
    from collections import Counter
    class_counts = Counter(labels)
    if categories is not None:
        nums = [class_counts.get(i, 0) for i in range(len(categories))]
        x_labels = categories
    else:
        x_labels = sorted(class_counts.keys())
        nums = [class_counts[k] for k in x_labels]
    plt.figure(figsize=(12, 6))
    plt.bar(x_labels, nums, color='skyblue')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Plot class distribution for CNN splits
def plot_cnn_class_distribution(counts, title, categories=None, save_path=None):
    if categories is None:
        categories = list(counts.keys())
    nums = [counts.get(label, 0) for label in categories]
    plt.figure(figsize=(12, 6))
    plt.bar(categories, nums, color='skyblue')
    plt.title(title)
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=90)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

# Transformation function
def apply_transformations_and_save(data, labels, output_dir, transform):
    """
    Apply a torchvision transform to each image and save to output_dir/class_{label}/img_{idx}_transformed.png.
    """
    import torch  # Ensure torch is imported
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, (img_array, label) in enumerate(zip(data, labels)):
        img = Image.fromarray((img_array * 255).astype(np.uint8))
        transformed_img = transform(img)
        # Only convert to PIL Image if it's a tensor
        if isinstance(transformed_img, torch.Tensor):
            transformed_img = transforms.ToPILImage()(transformed_img)
        class_dir = output_dir / f"class_{label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        transformed_img.save(class_dir / f"img_{idx}_transformed.png")

# CNN training functions

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
    plt.show()

def plot_cluster_scatter(features_2d, cluster_labels, title, save_path):
    """
    Plot a scatter plot of clustered features

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
    plt.show()

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
    plt.show()

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
    plt.show()

    print(f"Saved final optimized loss curve to {save_path}")

# Function to evaluate and visualize the CNN model
def evaluate_and_visualize_model(model, functional_model, val_generator, history, class_names=None):
    print("Starting evaluation and visualization...")
    output_dir = Path(config.CNN_RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")

    plot_training_curves(history, output_dir)
    y_true, y_pred, y_pred_probs = generate_predictions(functional_model, val_generator)
    if y_true is None:
        return

    plot_confusion_matrix(y_true, y_pred, class_names, output_dir)
    visualize_predictions(functional_model, val_generator, class_names, output_dir)
    generate_gradcam_visualizations(functional_model, val_generator, output_dir)


def plot_training_curves(history, output_dir):
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
    print("Training curves saved.")
    plt.show()


def generate_predictions(functional_model, val_generator):
    print("Generating predictions...")
    val_generator.reset()
    try:
        y_true = val_generator.classes
        y_pred_probs = functional_model.predict(val_generator)
        y_pred = np.argmax(y_pred_probs, axis=1)
        print("Predictions completed.")
        return y_true, y_pred, y_pred_probs
    except Exception as e:
        print(f"Failed during prediction: {e}")
        return None, None, None


def plot_confusion_matrix(y_true, y_pred, class_names, output_dir):
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    if class_names is not None:
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    else:
        cm_df = pd.DataFrame(cm)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrix.png')
    print("Confusion matrix saved.")
    plt.show()


def visualize_predictions(functional_model, val_generator, class_names, output_dir):
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
        pred_name = class_names[pred_label] if class_names else str(pred_label)
        true_name = class_names[true_label] if class_names else str(true_label)
        plt.title(f"{pred_name} {pred_probs[pred_label]*100:.1f}%\nTrue: {true_name}", color=color)
        plt.subplot(4, 4, i * 2 + 2)
        bars = plt.bar(range(len(class_names)), pred_probs, color='gray')
        bars[pred_label].set_color('green')
        plt.xticks(range(len(class_names)), [f"class_{i}" for i in range(len(class_names))], rotation=45)
        plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'predictions_grid.png')
    print("Prediction visualizations saved.")
    plt.show()


def generate_gradcam_visualizations(functional_model, val_generator, output_dir, return_overlays=False):
    print("Generating Grad-CAM visualizations...")
    gradcam_dir = output_dir / 'gradcam'
    gradcam_dir.mkdir(exist_ok=True)

    conv_layers = [layer for layer in functional_model.layers if isinstance(layer, Conv2D)]
    if not conv_layers:
        print("No Conv2D layers found for Grad-CAM.")
        return [] if return_overlays else None
    target_layer = conv_layers[-1]
    print(f"Using Conv2D layer for Grad-CAM: {target_layer.name}")

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
        heatmap = tf.maximum(heatmap, 0)
        percentile_99 = tfp.stats.percentile(heatmap, 99.0)
        heatmap = tf.clip_by_value(heatmap, 0, percentile_99)
        heatmap = heatmap / (percentile_99 + 1e-8)
        return heatmap.numpy()

    x_val, _ = next(val_generator)
    indices = np.random.choice(len(x_val), size=10, replace=False)
    overlays = []
    for i, idx in enumerate(indices):
        img = x_val[idx]
        img_array = np.expand_dims(img, axis=0)
        heatmap = make_gradcam_heatmap(img_array)

        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
        else:
            heatmap = np.zeros_like(heatmap)

        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        if img.dtype == np.float32 and img.max() <= 1.0:
            orig_img = (img * 255).astype(np.uint8)
        elif img.dtype == np.uint8:
            orig_img = img
        else:
            orig_img = np.clip(img * 255, 0, 255).astype(np.uint8)

        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_color, 0.4, 0)
        cv2.imwrite(str(gradcam_dir / f'gradcam_overlay_{i}.png'), overlay)
        if return_overlays:
            overlays.append(overlay)

    print(f"[INFO] Grad-CAM visualizations saved to {gradcam_dir}")
    if return_overlays:
        return overlays

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

def save_cnn_metrics_to_json(
    output_dir,
    history,
    class_names,
    y_true,
    y_pred,
    y_pred_probs,
    tta_report,
    non_tta_report=None,
    y_true_non_tta=None,
    y_pred_non_tta=None,
    y_pred_probs_non_tta=None
):
    results = {
        "history": history.history,
        "class_names": class_names,
        "tta": {
            "y_true": y_true.tolist(),
            "y_pred": y_pred.tolist(),
            "y_pred_probs": y_pred_probs.tolist(),
            "classification_report": tta_report,
        }
    }
    if non_tta_report is not None:
        results["non_tta"] = {
            "y_true": y_true_non_tta.tolist() if y_true_non_tta is not None else None,
            "y_pred": y_pred_non_tta.tolist() if y_pred_non_tta is not None else None,
            "y_pred_probs": y_pred_probs_non_tta.tolist() if y_pred_probs_non_tta is not None else None,
            "classification_report": non_tta_report,
        }
    with open(Path(output_dir) / "cnn_metrics.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Saved CNN metrics (TTA and non-TTA) to {Path(output_dir) / 'cnn_tta_vs_non_tta.json'}")

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

# RF training functions

def plot_rf_confusion_matrix_train(y_true, y_pred, save_path, class_names=None):
    """
    Plots and saves the confusion matrix for RF predictions (training/validation).
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    if class_names is not None:
        xticklabels = class_names
        yticklabels = class_names
    else:
        xticklabels = np.unique(y_true)
        yticklabels = np.unique(y_true)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title('Confusion Matrix (Train/Validation)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()


def extract_and_pivot_gridsearch_train(grid_search):
    """
    Extracts GridSearchCV results, filters for default hyperparameters,
    and pivots to get the mean_test_score matrix for heatmap plotting.
    Returns the scores_matrix DataFrame.
    """
    results_df = pd.DataFrame(grid_search.cv_results_)
    filtered = results_df[
        (results_df.param_min_samples_split == 2) &
        (results_df.param_min_samples_leaf == 1) &
        (results_df.param_max_features == 'sqrt') &
        (results_df.param_class_weight == 'balanced') &
        (results_df.param_bootstrap == True)
    ]
    scores_matrix = filtered.pivot(index='param_n_estimators', columns='param_max_depth', values='mean_test_score')
    return scores_matrix

def plot_rf_heatmap_train(scores_matrix, save_path):
    """
    Plots and saves a heatmap of GridSearchCV accuracy for n_estimators vs max_depth.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(scores_matrix, annot=True, fmt=".4f", cmap='viridis')
    plt.title('GridSearchCV Accuracy: n_estimators vs max_depth')
    plt.xlabel('max_depth')
    plt.ylabel('n_estimators')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Heatmap saved to {save_path}")
    plt.show()

def plot_top20_gridsearch_models(sorted_scores, save_path, top_k=20):
    """
    Plots and saves the top 20 GridSearchCV results.
    """
    plt.figure(figsize=(14, 6))
    plt.plot(range(top_k), sorted_scores[:top_k], marker='o')
    plt.title('Top 20 GridSearchCV Results')
    plt.xlabel('Model Rank')
    plt.ylabel('Mean Cross-Validation Score')
    plt.grid(True)
    plt.xticks(range(top_k), [f"Model {i+1}" for i in range(top_k)], rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Top 20 models plot saved to {save_path}")
    plt.show()

def save_rf_gridsearch_metadata_train(
    grid_search,
    param_grid,
    confusion_matrix_path,
    top20_plot_path,
    heatmap_path,
    save_dir
):
    """
    Saves RF grid search metadata and paths to result images as a JSON file.
    """
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
    metadata_path = save_dir / 'gridsearch_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(results_metadata, f, indent=4)
    print(f"\nGridSearch metadata saved to {metadata_path}")
    plt.show()

# CNN test functions

def plot_prediction_explanation_grid(generator, y_pred_probs, y_pred_labels, class_names, save_path, n_images=10):
    """
    Plots a grid of images with their predicted probabilities bar chart.
    """
    filepaths = generator.filepaths
    true_labels = generator.classes
    indices = np.random.choice(len(filepaths), size=min(n_images, len(filepaths)), replace=False)
    n_cols = 2
    n_rows = int(np.ceil(n_images / n_cols))
    plt.figure(figsize=(n_cols * 6, n_rows * 3.5))
    for i, idx in enumerate(indices):
        img = plt.imread(filepaths[idx])
        pred_label = y_pred_labels[idx]
        true_label = true_labels[idx]
        pred_prob = y_pred_probs[idx]
        confidence = 100 * np.max(pred_prob)
        pred_class = class_names[pred_label]
        true_class = class_names[true_label]
        color = "green" if pred_label == true_label else "red"
        # Image
        plt.subplot(n_rows, n_cols * 2, i * 2 + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"{pred_class} {confidence:.1f}% (class_{pred_label})\nTrue: {true_class} (class_{true_label})", color=color, fontsize=11)
        # Bar chart
        plt.subplot(n_rows, n_cols * 2, i * 2 + 2)
        bars = plt.bar(range(len(class_names)), pred_prob, color=["green" if j == pred_label else "gray" for j in range(len(class_names))])
        plt.xticks(range(len(class_names)), range(len(class_names)))
        plt.xlabel("Class #")
        plt.ylim(0, 1)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
    plt.savefig(save_path)
    print(f"Prediction explanation grid saved to {save_path}")
    plt.show()

def plot_cnn_probabilities_heatmap_test(y_pred_probs, save_path, class_names=None):
    """
    Plots and saves a heatmap of predicted probabilities for the CNN model.
    """
    plt.figure(figsize=(12, 6))
    if class_names is not None:
        xticklabels = class_names
    else:
        xticklabels = [str(i) for i in range(y_pred_probs.shape[1])]
    sns.heatmap(y_pred_probs, cmap='viridis', xticklabels=xticklabels)
    plt.title('Test Set Prediction Probabilities Heatmap')
    plt.xlabel('Class')
    plt.ylabel('Sample')
    plt.savefig(save_path)
    print(f"Probabilities heatmap saved to {save_path}")
    plt.show()

def plot_confusion_matrix_cnn_test(y_true, y_pred, class_names, save_path):
    """
    Plots and saves the confusion matrix for CNN predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()

# RF test functions

def plot_rf_feature_importance_test(rf_model, feature_names, save_path, top_n=20):
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.title("Top Feature Importances")
    plt.bar(range(top_n), importances[indices], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Feature importance plot saved to {save_path}")
    plt.show()

def plot_rf_probabilities_heatmap_test(proba, save_path, class_names=None):
    """
    Plots and saves a heatmap of predicted probabilities for the RF model.
    """
    plt.figure(figsize=(12, 6))
    if class_names is not None:
        xticklabels = class_names
    else:
        xticklabels = [str(i) for i in range(proba.shape[1])]
    sns.heatmap(proba, cmap='viridis', xticklabels=xticklabels)
    plt.title('Test Set Prediction Probabilities Heatmap')
    plt.xlabel('Class')
    plt.ylabel('Sample')
    plt.savefig(save_path)
    print(f"Probabilities heatmap saved to {save_path}")
    plt.show()

def plot_rf_confusion_matrix_test(y_true, y_pred, save_path, class_names=None):
    """
    Plots and saves the confusion matrix for RF predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    if class_names is not None:
        xticklabels = class_names
        yticklabels = class_names
    else:
        xticklabels = np.unique(y_true)
        yticklabels = np.unique(y_true)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=xticklabels, yticklabels=yticklabels)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")
    plt.show()