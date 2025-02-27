"""
Test script for the line grouping data loader
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import random

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from objects.data_loader import ReceiptLineDataset

data_dir = os.path.join(project_root, 'preprocessed_data')
img_height = 32
max_width = 320


transform = transforms.Compose([
    transforms.Resize((img_height, max_width)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

line_height_thresholds = [5, 10, 15, 20]

for threshold in line_height_thresholds:
    print(f"\n=== Testing with line height threshold: {threshold} ===")
    
    try:
        val_dataset = ReceiptLineDataset(
            root_dir=data_dir,
            split='valid',
            transform=None,  # No transform for visualization
            line_height_threshold=threshold,
            min_line_boxes=3  # Minimum boxes to form a valid line
        )
        
        print(f"Found {len(val_dataset)} valid line samples with threshold {threshold}")
        
        if len(val_dataset) > 0:
            # Get a random sample from the dataset
            sample_idx = random.randint(0, len(val_dataset) - 1)
            line_region, target, text, line_boxes = val_dataset[sample_idx]
            
            img_path = val_dataset.samples[sample_idx][0]
            original_image = Image.open(img_path).convert('L')
            
            # Create a figure with 3 subplots
            fig, ax = plt.subplots(3, 1, figsize=(12, 12))
            
            # 1. Original image with all bounding boxes
            ax[0].imshow(original_image, cmap='gray')
            ax[0].set_title(f"Original Image with Line Grouping (threshold={threshold})")
            
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            txt_path = os.path.join(os.path.dirname(img_path), f"{base_name}.txt")
            all_bboxes = val_dataset.parse_bbox_file(txt_path)
            
            for x, y, w, h in all_bboxes:
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='gray', facecolor='none')
                ax[0].add_patch(rect)
            
            all_lines = val_dataset.group_boxes_into_lines(all_bboxes)
            
            colors = ['r', 'g', 'b', 'c', 'm', 'y']
            for i, line in enumerate(all_lines):
                line_color = colors[i % len(colors)]
                
                for x, y, w, h in line:
                    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor=line_color, facecolor='none')
                    ax[0].add_patch(rect)
                
                if line:
                    min_x = min(box[0] for box in line)
                    min_y = min(box[1] for box in line)
                    max_x = max(box[0] + box[2] for box in line)
                    max_y = max(box[1] + box[3] for box in line)
                    
                    rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                            linewidth=2, edgecolor=line_color, facecolor='none', 
                                            linestyle='--')
                    ax[0].add_patch(rect)
            
            # 2. Selected line region with boxes
            ax[1].imshow(original_image, cmap='gray')
            ax[1].set_title(f"Selected Line (sample {sample_idx})")
            
            for x, y, w, h in line_boxes:
                rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor='r', facecolor='none')
                ax[1].add_patch(rect)
            
            if line_boxes:
                min_x = min(box[0] for box in line_boxes)
                min_y = min(box[1] for box in line_boxes)
                max_x = max(box[0] + box[2] for box in line_boxes)
                max_y = max(box[1] + box[3] for box in line_boxes)
                
                rect = patches.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y, 
                                        linewidth=2, edgecolor='g', facecolor='none', 
                                        linestyle='--')
                ax[1].add_patch(rect)
            
            # 3. Extracted line region
            ax[2].imshow(line_region, cmap='gray')
            ax[2].set_title(f"Extracted Line Region (synthetic text: '{text}')")
            
            # Save the visualization
            test_images_dir = os.path.join(project_root, 'test_images')
            os.makedirs(test_images_dir, exist_ok=True)
            fig_path = os.path.join(test_images_dir, f'line_grouping_threshold_{threshold}.png')
            plt.tight_layout()
            plt.savefig(fig_path)
            print(f"Visualization saved to {fig_path}")
        
    except Exception as e:
        print(f"Error with threshold {threshold}: {str(e)}")
        import traceback
        traceback.print_exc()

results = []
for threshold in [5, 10, 15, 20, 30, 40]:
    try:
        dataset = ReceiptLineDataset(
            root_dir=data_dir,
            split='valid',
            transform=None,
            line_height_threshold=threshold,
            min_line_boxes=3
        )
        results.append((threshold, len(dataset)))
    except Exception as e:
        print(f"Error with threshold {threshold}: {str(e)}")
        results.append((threshold, 0))

print("\n=== Line Count with Different Thresholds ===")
for threshold, count in results:
    print(f"Threshold {threshold}: {count} lines")

# Visualize how many lines we get with different thresholds
plt.figure(figsize=(10, 6))
thresholds, counts = zip(*results)
plt.plot(thresholds, counts, 'o-')
plt.xlabel('Line Height Threshold')
plt.ylabel('Number of Lines Detected')
plt.title('Effect of Line Height Threshold on Line Detection')
plt.grid(True)

plt.savefig(os.path.join(test_images_dir, 'threshold_vs_linecount.png'))