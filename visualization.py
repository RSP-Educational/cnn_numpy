import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_images(images:np.ndarray, labels:np.ndarray, num_images:int = 10, n_cols:int = 8):
    """Plot a grid of images with their corresponding labels.

    Parameters
    ----------
    images : np.ndarray
        Array of shape (N, H, W) containing the images to plot.

    labels : np.ndarray
        Array of shape (N,) containing the labels corresponding to the images.

    num_images : int
        Number of images to plot. Default is 10.
    """
    plt.figure(figsize=(10, 10))
    for i in range(len(images)):
        if i >= num_images:
            break
        plt.subplot((num_images + n_cols - 1) // n_cols, n_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'Label: {labels[i]}')
        plt.axis('off')
    plt.show()

