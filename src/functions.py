# Basic libraries
import os
import random
import collections
from pathlib import Path

# Image and array libraries
import numpy as np
from PIL import Image
import cv2

# Plotting libraries
import matplotlib.pyplot as plt

# Progress bars
from tqdm import tqdm

# Deep learning libraries
from tensorflow.keras import layers, models

# Project-specific config
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
        plt.imshow(img)
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
    autoencoder = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(2048, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(512, activation='relu'),
        layers.Dense(bottleneck_dim, activation='relu', name='bottleneck'),
        layers.Dropout(0.2),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(2048, activation='relu'),
        layers.Dense(input_dim, activation='sigmoid')
    ])
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
