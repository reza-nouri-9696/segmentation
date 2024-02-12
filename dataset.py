import os
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform_img=None, transform_label=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
        self.transform_img = transform_img
        self.transform_label = transform_label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        # Convert image to NumPy array
        img_array = np.array(img)

        # Split the image array into image and label parts
        image, label = img_array[:, :img_array.shape[1] // 2], img_array[:, img_array.shape[1] // 2:]

        if self.transform_img:
            image = self.transform_img(image)

        if self.transform_label:
            label = self.transform_label(label)

        return image, label
