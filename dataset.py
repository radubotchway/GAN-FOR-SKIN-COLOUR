#dataset.py
import os
from PIL import Image
from torch.utils.data import Dataset  # Import Dataset class from torch.utils.data

class SkinDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))

        # Initialize a list to hold all image paths and corresponding class indices
        self.image_paths = []
        self.class_indices = []

        # Iterate through each class directory and collect image paths
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                self.image_paths.append(img_path)
                self.class_indices.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Handle index out of range gracefully
        if idx >= len(self.image_paths):
            raise IndexError(f"Index {idx} is out of range for dataset with {len(self.image_paths)} samples.")

        img_path = self.image_paths[idx]
        class_idx = self.class_indices[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, class_idx
