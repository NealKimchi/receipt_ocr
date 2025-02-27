"""
Test script with sample data generation
"""

import os
import sys
import json
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import random

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# First, create the sample_annotations.json file directly
def create_sample_annotations():
    """Create a sample annotations file with manually defined text and boxes."""
    
    print("Creating sample annotations file...")
    
    # Define a list of common receipt items
    items = [
        {"item_name": "Nasi Campur Bali", "item_quantity": "1", "item_value": "75,000"},
        {"item_name": "Bbk Bengil Nasi", "item_quantity": "1", "item_value": "125,000"},
        {"item_name": "MilkShake Strawberry", "item_quantity": "1", "item_value": "37,000"},
        {"item_name": "Ice Lemon Tea", "item_quantity": "1", "item_value": "24,000"},
        {"item_name": "Nasi Ayam Dewata", "item_quantity": "1", "item_value": "70,000"},
        {"item_name": "Free Ice Tea", "item_quantity": "3", "item_value": "0"},
        {"item_name": "Organic Green Salad", "item_quantity": "1", "item_value": "65,000"},
        {"item_name": "Ice Tea", "item_quantity": "1", "item_value": "18,000"},
        {"item_name": "Ice Orange", "item_quantity": "1", "item_value": "29,000"},
        {"item_name": "Ayam Suir Bali", "item_quantity": "1", "item_value": "85,000"}
    ]
    
    # Sample bounding boxes for these items
    boxes = [
        [300, 367, 197, 23],  # Nasi Campur Bali
        [543, 362, 74, 23],   # 75,000
        [304, 394, 182, 23],  # Bbk Bengil Nasi
        [533, 388, 85, 24],   # 125,000
        [304, 420, 194, 23],  # MilkShake Strawberry
        [541, 413, 77, 26],   # 37,000
        [306, 447, 160, 23],  # Ice Lemon Tea
        [544, 440, 74, 24],   # 24,000
        [306, 474, 195, 23],  # Nasi Ayam Dewata
        [543, 466, 75, 24],   # 70,000
    ]
    
    # Create a parsed data structure
    parsed_data = {
        "line_items": items,
        "subtotal": {
            "subtotal": "1,346,000",
            "service": "100,950",
            "tax": "144,695"
        },
        "total": {
            "total": "1,591,600"
        }
    }
    
    # Create OCR words string
    words = [item["item_name"] for item in items] + [item["item_value"] for item in items]
    ocr_words = str(words)
    
    # Create OCR boxes string (simplified format)
    ocr_boxes = []
    for i, box in enumerate(boxes):
        x, y, w, h = box
        box_str = f"[[[[{x}, {y}], [{x+w}, {y}], [{x+w}, {y+h}], [{x}, {y+h}]]], ('text_{i}', 0.95)]"
        ocr_boxes.append(box_str)
    ocr_boxes_str = "[" + ", ".join(ocr_boxes) + "]"
    
    # Create annotations dictionary
    annotations = {}
    
    # Add annotations for test images
    valid_dir = os.path.join(project_root, 'preprocessed_data', 'valid')
    if os.path.exists(valid_dir):
        print(f"Adding annotations for images in: {valid_dir}")
        image_files = [f for f in os.listdir(valid_dir) if f.endswith(('.jpg', '.png'))]
        for img_file in image_files:
            print(f"  Adding annotation for: {img_file}")
            annotations[img_file] = {
                "ocr_words": ocr_words,
                "ocr_boxes": ocr_boxes_str,
                "parsed_data": json.dumps(parsed_data)
            }
    else:
        # Just add one sample entry
        print("No valid directory found, adding a sample annotation")
        annotations["sample_receipt.jpg"] = {
            "ocr_words": ocr_words,
            "ocr_boxes": ocr_boxes_str,
            "parsed_data": json.dumps(parsed_data)
        }
    
    # Save to file
    annotations_file = os.path.join(project_root, 'sample_annotations.json')
    with open(annotations_file, 'w', encoding='utf-8') as f:
        json.dump(annotations, f, indent=2)
    
    print(f"Created sample annotations file at {annotations_file}")
    return annotations_file


def create_valid_samples_json():
    """Create a valid_samples.json file with sample text lines."""
    
    print("Creating valid_samples.json file...")
    
    # Sample text lines
    text_lines = [
        "Nasi Campur Bali 75,000",
        "Bbk Bengil Nasi 125,000",
        "MilkShake Strawberry 37,000", 
        "Ice Lemon Tea 24,000",
        "Nasi Ayam Dewata 70,000",
        "Free Ice Tea 0",
        "Organic Green Salad 65,000",
        "Ice Tea 18,000",
        "Ice Orange 29,000",
        "Ayam Suir Bali 85,000",
        "Subtotal 1,346,000",
        "Service 100,950",
        "Tax 144,695",
        "Total 1,591,600"
    ]
    
    # Create sample entries
    samples = []
    for i, text in enumerate(text_lines):
        # Create a sample with dummy paths that will be replaced by actual data
        sample = {
            "image_path": f"dummy_path_{i}.jpg",
            "line_box": [100, 100 + i*30, 400, 25],
            "text": text
        }
        samples.append(sample)
    
    # Save to the valid_samples.json file
    valid_dir = os.path.join(project_root, 'preprocessed_data')
    os.makedirs(valid_dir, exist_ok=True)
    
    valid_samples_file = os.path.join(valid_dir, 'valid_samples.json')
    with open(valid_samples_file, 'w', encoding='utf-8') as f:
        json.dump(samples, f, indent=2)
    
    print(f"Created {len(samples)} sample entries in {valid_samples_file}")
    return valid_samples_file


def generate_sample_line_images():
    """Generate sample text line images for testing."""
    
    print("Generating sample text line images...")
    
    # Sample text lines
    text_lines = [
        "Nasi Campur Bali 75,000",
        "Bbk Bengil Nasi 125,000",
        "MilkShake Strawberry 37,000", 
        "Ice Lemon Tea 24,000",
        "Nasi Ayam Dewata 70,000",
        "Free Ice Tea 0",
        "Organic Green Salad 65,000",
        "Ice Tea 18,000",
        "Ice Orange 29,000",
        "Ayam Suir Bali 85,000",
        "Subtotal 1,346,000",
        "Service 100,950",
        "Tax 144,695",
        "Total 1,591,600"
    ]
    
    # Create a directory for the sample images
    line_regions_dir = os.path.join(project_root, 'line_regions')
    os.makedirs(line_regions_dir, exist_ok=True)
    
    # Generate an image for each text line
    for i, text in enumerate(text_lines):
        # Create a blank image
        image = Image.new('L', (400, 30), color=255)
        draw = ImageDraw.Draw(image)
        
        # Add text to the image
        try:
            font = ImageFont.truetype("Arial.ttf", 16)
        except IOError:
            # If Arial not available, use default font
            font = ImageFont.load_default()
            
        draw.text((10, 5), text, fill=0, font=font)
        
        # Save the image
        image_path = os.path.join(line_regions_dir, f"line_{i}.png")
        image.save(image_path)
    
    print(f"Generated {len(text_lines)} sample text line images in {line_regions_dir}")
    
    # Create a visualization of all lines
    fig, axes = plt.subplots(len(text_lines), 1, figsize=(10, 2*len(text_lines)))
    
    for i, text in enumerate(text_lines):
        image_path = os.path.join(line_regions_dir, f"line_{i}.png")
        image = Image.open(image_path)
        axes[i].imshow(np.array(image), cmap='gray')
        axes[i].set_title(f"Line {i+1}: {text}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(project_root, 'line_images.png'))
    print(f"Created visualization at {os.path.join(project_root, 'line_images.png')}")


if __name__ == "__main__":
    print("=== Starting Sample Data Generation ===")
    
    # Create sample annotations
    create_sample_annotations()
    
    # Create valid_samples.json
    create_valid_samples_json()
    
    # Generate sample line images
    generate_sample_line_images()
    
    print("=== Sample Data Generation Complete ===")
    print("Now you can run the test_annotations.py script to test the model")