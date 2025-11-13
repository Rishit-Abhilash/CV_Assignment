# Memory-Efficient Data Loading Script
# Use this instead of np.load() to avoid memory errors

import sys
sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# Import config
exec(open('00_utils_and_config.ipynb').read())

# Memory-mapped dataset class
class MemoryMappedDataset(Dataset):
    """Memory-efficient dataset using memory-mapped numpy arrays."""

    def __init__(self, X_path, y_path, transform=None, normalize=True):
        """
        Args:
            X_path: Path to .npy file with images
            y_path: Path to .npy file with labels
            transform: Optional transform to apply
            normalize: Whether to normalize to [0, 1]
        """
        # Load labels (small, can fit in memory)
        self.labels = np.load(y_path)

        # Memory-map the image data (doesn't load into RAM)
        self.images = np.load(X_path, mmap_mode='r')

        self.transform = transform
        self.normalize = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Load single image from memory-mapped file
        image = np.array(self.images[idx])  # Load only this one image
        label = self.labels[idx]

        # Normalize if needed
        if self.normalize and image.dtype == np.uint8:
            image = image.astype('float32') / 255.0

        # Convert to PyTorch format (C, H, W) if needed
        if len(image.shape) == 3 and image.shape[-1] in [1, 3]:
            # (H, W, C) -> (C, H, W)
            image = np.transpose(image, (2, 0, 1))

        # Convert to tensor
        image = torch.from_numpy(image).float()

        return image, torch.tensor(label, dtype=torch.long)


print("Memory-efficient dataset class loaded!")
print("\nExample usage:")
print("  train_dataset = MemoryMappedDataset(")
print("      X_path='../processed_data/X_train_224.npy',")
print("      y_path='../processed_data/y_train.npy',")
print("      normalize=True")
print("  )")
