import torch
import numpy as np
import os
from PIL import Image

# --- UTILITY: Intersection over Union (IoU) ---
def intersection_over_union(box1, box2, box_format="midpoint"):
    """
    Calculates intersection over union. Assumes inputs are (N, 4) tensors.
    """
    # ----------------------------------------------------
    # --- BLOCK 1: Define variables based on box_format ---
    # ----------------------------------------------------
    if box_format == "midpoint":
        # CONVERT [x_c, y_c, w, h] to [x1, y1, x2, y2]
        box1_x1 = box1[..., 0:1] - box1[..., 2:3] / 2
        box1_y1 = box1[..., 1:2] - box1[..., 3:4] / 2
        box1_x2 = box1[..., 0:1] + box1[..., 2:3] / 2
        box1_y2 = box1[..., 1:2] + box1[..., 3:4] / 2
        
        box2_x1 = box2[..., 0:1] - box2[..., 2:3] / 2
        box2_y1 = box2[..., 1:2] - box2[..., 3:4] / 2
        box2_x2 = box2[..., 0:1] + box2[..., 2:3] / 2
        box2_y2 = box2[..., 1:2] + box2[..., 3:4] / 2
    
    elif box_format == "corners":
        # INPUT IS ALREADY [x1, y1, x2, y2]
        box1_x1 = box1[..., 0:1]
        box1_y1 = box1[..., 1:2]
        box1_x2 = box1[..., 2:3]
        box1_y2 = box1[..., 3:4]

        box2_x1 = box2[..., 0:1]
        box2_y1 = box2[..., 1:2]
        box2_x2 = box2[..., 2:3]
        box2_y2 = box2[..., 3:4]

    else:
        # Handle unexpected format if necessary
        raise ValueError("Invalid box_format specified.")
        
    # ----------------------------------------------------
    # --- BLOCK 2: Calculate Intersection and Union (UNCHANGED) ---
    # ----------------------------------------------------

    # Find coordinates of intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Intersection area: clamp(0) ensures non-intersecting boxes result in 0 area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    
    # Union area
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + 1e-6 # Add epsilon for stability

    return intersection / union

# --- CUSTOM DATASET CLASS (Converts YOLO labels to Grid Tensor) ---
class CustomYOLODataset(torch.utils.data.Dataset):
    def __init__(self, data_path, S=7, B=1, C=2, transform=None):
        self.image_dir = os.path.join(data_path, 'images')
        self.label_dir = os.path.join(data_path, 'labels')
        self.transform = transform
        
        self.image_files = [f for f in os.listdir(self.image_dir) if f.endswith('.jpg')]
        
        self.S = S  # Grid Size
        self.B = B  # Boxes per cell
        self.C = C  # Classes (e.g., person, car)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        label_name = img_name.replace('.jpg', '.txt')
        
        img_path = os.path.join(self.image_dir, img_name)
        label_path = os.path.join(self.label_dir, label_name)
        
        image = Image.open(img_path).convert("RGB")
        
        boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    # Class x_c y_c w h are normalized to 0-1
                    class_id, x_c, y_c, w, h = map(float, line.strip().split())
                    boxes.append([class_id, x_c, y_c, w, h])

        if self.transform:
            image = self.transform(image) 
        
        # KEY: Convert normalized boxes to the 7x7 Grid Target Tensor
        target_tensor = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_id, x_c, y_c, w, h = box
            class_id = int(class_id)
            
            # i, j are the grid cell coordinates (row, column)
            i = int(self.S * y_c)
            j = int(self.S * x_c)

            i = min(i, self.S - 1)
            j = min(j, self.S - 1)
            
            # x_cell, y_cell are coords relative to the cell (0 to 1)
            x_cell = self.S * x_c - j
            y_cell = self.S * y_c - i

            if target_tensor[i, j, 4] == 0:
                target_tensor[i, j, 4] = 1.0 # Set confidence
                target_tensor[i, j, 0:4] = torch.tensor([x_cell, y_cell, w, h])
                target_tensor[i, j, 5 + class_id] = 1.0 # One-hot class encoding

        return image, target_tensor
