import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import random
import albumentations as A
from tqdm.notebook import tqdm
import os

class ReceiptPreprocessor:
    def __init__(self, dataset, output_dir="preprocessed_data"):
        """
        Initialize the receipt preprocessor.
        Args:
            output_dir (str): Directory to save processed images
        """
        import os
        self.dataset = dataset
        if not os.path.isabs(output_dir):
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_file_dir)  # Go up to project root
            self.output_dir = os.path.join(project_root, output_dir)
        else:
            self.output_dir = output_dir
            
        print(f"Setting output directory to: {self.output_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "valid"), exist_ok=True)
        
        # Define augmentation pipeline
        self.augmentation = A.Compose([
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.Rotate(limit=5, p=0.5),
                A.Affine(scale=(0.95, 1.05), translate_percent=0.05, p=0.5),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2.0, p=0.5),
                A.Sharpen(alpha=(0.1, 0.3), p=0.5),
            ], p=0.5),
        ])
    
    def preprocess_pil(self, image, enhance=True):
        """
        Preprocess an image using PIL operations.
        
        Args:
            image (PIL.Image): Input image
            enhance (bool): Whether to apply enhancement operations
            
        Returns:
            PIL.Image: Preprocessed image
        """
        # Convert to grayscale for document processing
        if image.mode != 'L':
            image = image.convert('L')
        
        if enhance:
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.5)
            
            # Apply slight sharpening
            image = image.filter(ImageFilter.SHARPEN)
            
            # Apply adaptive thresholding (using custom method)
            image = self._adaptive_threshold_pil(image)
        
        return image
    
    def _adaptive_threshold_pil(self, image, block_size=11, c=2):
        """
        Apply adaptive thresholding to PIL image.
        
        Args:
            image (PIL.Image): Input image
            block_size (int): Size of the local neighborhood for thresholding
            c (float): Constant subtracted from the mean
            
        Returns:
            PIL.Image: Thresholded image
        """
        # Convert PIL to OpenCV format for thresholding
        cv_img = np.array(image)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            cv_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, block_size, c
        )
        
        # Convert back to PIL
        return Image.fromarray(thresh)
    
    def preprocess_cv2(self, image_array):
        """
        Preprocess an image using OpenCV operations.
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Dilation to enhance text
        kernel = np.ones((1, 1), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        return dilated
    
    def detect_text_regions(self, image_array, min_area=100, max_area_ratio=0.2, min_aspect=0.2, max_aspect=5):
        """
        Detect potential text regions in an image with better filtering.
        
        Args:
            image_array (numpy.ndarray): Input image as numpy array
            min_area (int): Minimum contour area to consider
            max_area_ratio (float): Maximum area as ratio of image size
            min_aspect, max_aspect (float): Valid aspect ratio range for text
            
        Returns:
            list: List of bounding boxes [(x, y, w, h), ...]
        """
        # Ensure image is in the right format
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_array
        
        # Get image dimensions
        height, width = gray.shape
        max_area = height * width * max_area_ratio
        
        # Preprocessing for better contour detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        bounding_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / float(h) if h > 0 else 0
            # Skip thin horizontal lines (likely separators)
            if h <= 3 and w > width * 0.1:
                continue
            # Skip thin vertical lines (likely borders)
            if w <= 3 and h > height * 0.1:
                continue
            
            if (min_area < area < max_area and 
                min_aspect < aspect_ratio < max_aspect):
                bounding_boxes.append((x, y, w, h))
        
        return bounding_boxes
    
    def merge_nearby_boxes(self, boxes, threshold=10):
        """
        Merge nearby bounding boxes that likely belong to the same text region.
        
        Args:
            boxes (list): List of bounding boxes [(x, y, w, h), ...]
            threshold (int): Distance threshold for merging boxes
            
        Returns:
            list: Merged bounding boxes
        """
        if not boxes:
            return []
        
        boxes_xyxy = [(x, y, x + w, y + h) for x, y, w, h in boxes]
        
        boxes_xyxy.sort(key=lambda box: box[1])
        
        merged_boxes = []
        current_box = list(boxes_xyxy[0])
        
        for box in boxes_xyxy[1:]:
            # If boxes are close vertically and overlap horizontally
            if (abs(box[1] - current_box[1]) < threshold and 
                max(0, min(current_box[2], box[2]) - max(current_box[0], box[0])) > 0):
                current_box[0] = min(current_box[0], box[0])
                current_box[1] = min(current_box[1], box[1])
                current_box[2] = max(current_box[2], box[2])
                current_box[3] = max(current_box[3], box[3])
            else:
                merged_boxes.append(tuple(current_box))
                current_box = list(box)
        
        merged_boxes.append(tuple(current_box))
        
        return [(x, y, x2 - x, y2 - y) for x, y, x2, y2 in merged_boxes]
    
    def apply_augmentation(self, image_array, boxes=None):
        """
        Apply augmentation to an image and adjust bounding boxes accordingly.
        
        Args:
            image_array (numpy.ndarray): Input image
            boxes (list, optional): List of bounding boxes
            
        Returns:
            tuple: (augmented_image, transformed_boxes)
        """
        if boxes is None or len(boxes) == 0:
            # Just transform the image without bounding boxes
            augmented = self.augmentation(image=image_array)
            return augmented['image'], []
        
        # Convert boxes from (x, y, w, h) to albumentations format [x1, y1, x2, y2]
        bboxes = [[x, y, x + w, y + h] for x, y, w, h in boxes]
        
        # Convert list to numpy array for albumentations
        import numpy as np
        bboxes = np.array(bboxes, dtype=np.float32)
        
        # Apply augmentation with bounding boxes
        try:
            augmented = self.augmentation(
                image=image_array,
                bboxes=bboxes,
                # Add a dummy label for each box
                bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']),
                labels=['text'] * len(bboxes)
            )
            
            # Convert back to (x, y, w, h) format
            transformed_boxes = []
            for bbox in augmented['bboxes']:
                x1, y1, x2, y2 = bbox
                transformed_boxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            
            return augmented['image'], transformed_boxes
        
        except Exception as e:
            print(f"Warning: Augmentation failed with error: {str(e)}")
            print("Returning original image and boxes")
            return image_array, boxes
    
    def prepare_dataset(self, split="train", num_samples=None, augment=True):
        """
        Process a batch of images from the dataset.
        
        Args:
            dataset: The dataset object
            split (str): Dataset split ('train', 'val', etc.)
            num_samples (int, optional): Number of samples to process
            augment (bool): Whether to apply augmentation
            
        Returns:
            dict: Stats about processed images
        """
        if num_samples is None:
            num_samples = len(self.dataset[split])
        else:
            num_samples = min(num_samples, len(self.dataset[split]))
            
        os.makedirs(os.path.join(self.output_dir, split), exist_ok=True)
        
        stats = {
            'total_processed': 0,
            'total_boxes_detected': 0,
            'images_with_boxes': 0
        }
        
        for i in tqdm(range(num_samples), desc=f"Processing {split} images"):
            sample = self.dataset[split][i]
            
            # Extract image
            if 'image' in sample:
                image = sample['image']
                if not isinstance(image, Image.Image):
                    if isinstance(image, bytes):
                        image = Image.open(io.BytesIO(image))
                    elif isinstance(image, np.ndarray):
                        image = Image.fromarray(image)
            else:
                print(f"No image found in sample {i}")
                continue
            
            # Convert to numpy array for processing
            image_array = np.array(image)
            
            # Detect text regions
            boxes = self.detect_text_regions(image_array)
            
            # Merge nearby boxes
            merged_boxes = self.merge_nearby_boxes(boxes)
            
            # Apply augmentation if needed
            if augment and split == "train":
                # Create multiple augmented versions
                num_augmentations = 3
                
                for aug_idx in range(num_augmentations):
                    aug_image, aug_boxes = self.apply_augmentation(
                        image_array, merged_boxes
                    )
                    
                    # Save the augmented image
                    output_path = os.path.join(
                        self.output_dir, split, 
                        f"sample_{i}_aug{aug_idx}.jpg"
                    )
                    Image.fromarray(aug_image).save(output_path)
                    
                    # Save bounding box annotations
                    self.save_annotations(
                        sample.get('id', i), aug_idx, aug_boxes, split
                    )
            else:
                # Just save the processed image without augmentation
                processed = self.preprocess_cv2(image_array)
                output_path = os.path.join(
                    self.output_dir, split, f"{sample.get('id', i)}.jpg"
                )
                Image.fromarray(processed).save(output_path)
                
                # Save bounding box annotations
                self.save_annotations(
                    sample.get('id', i), None, merged_boxes, split
                )
            
            # Update stats
            stats['total_processed'] += 1
            stats['total_boxes_detected'] += len(merged_boxes)
            if len(merged_boxes) > 0:
                stats['images_with_boxes'] += 1
        
        return stats
    
    def save_annotations(self, sample_id, aug_idx, boxes, split):
        """
        Save bounding box annotations to a file.
        
        Args:
            sample_id: Identifier for the sample
            aug_idx: Augmentation index or None
            boxes: List of bounding boxes
            split: Dataset split
        """
        filename = f"{sample_id}"
        if aug_idx is not None:
            filename += f"_aug{aug_idx}"
        
        annotation_path = os.path.join(
            self.output_dir, split, f"{filename}.txt"
        )
        
        with open(annotation_path, 'w') as f:
            for box in boxes:
                # Format: x,y,width,height
                f.write(f"{box[0]},{box[1]},{box[2]},{box[3]}\n")
    
    def visualize_processed_image(self, sample_idx, split="train", show_boxes=True):
        """
        Visualize a processed image with detected text regions.
        
        Args:
            sample_idx (int): Index of the sample to visualize
            split (str): Dataset split
            show_boxes (bool): Whether to display bounding boxes
        """
        sample = self.dataset[split][sample_idx]
        
        # Get the original image
        if 'image' in sample:
            image = sample['image']
            if not isinstance(image, Image.Image):
                if isinstance(image, bytes):
                    image = Image.open(io.BytesIO(image))
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(image)
        else:
            print(f"No image found in sample {sample_idx}")
            return
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Process the image using OpenCV methods
        processed_cv2 = self.preprocess_cv2(image_array)
        
        # Process the image using PIL methods
        pil_image = Image.fromarray(image_array)
        processed_pil = self.preprocess_pil(pil_image)
        processed_pil_array = np.array(processed_pil)
        
        # Detect text regions
        boxes = self.detect_text_regions(image_array)
        merged_boxes = self.merge_nearby_boxes(boxes)
        
        # Create augmented version
        augmented, aug_boxes = self.apply_augmentation(image_array, merged_boxes)
        
        # Prepare images for visualization
        images = [
            ("Original", image_array),
            ("OpenCV Processed", processed_cv2),
            ("PIL Processed", processed_pil_array),
            ("Augmented", augmented)
        ]
        
        # Set up the figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, (title, img) in enumerate(images):
            # Display the image
            if len(img.shape) == 2:
                axes[i].imshow(img, cmap='gray')
            else:
                axes[i].imshow(img)
            
            axes[i].set_title(title)
            axes[i].axis('off')
            
            # Draw bounding boxes if requested
            if show_boxes:
                boxes_to_draw = merged_boxes
                if title == "Augmented":
                    boxes_to_draw = aug_boxes
                
                for x, y, w, h in boxes_to_draw:
                    rect = plt.Rectangle(
                        (x, y), w, h, 
                        fill=False, edgecolor='red', linewidth=2
                    )
                    axes[i].add_patch(rect)
        
        plt.tight_layout()
        plt.show()
        
        # Print stats
        print(f"Original detected boxes: {len(boxes)}")
        print(f"Merged boxes: {len(merged_boxes)}")
        if aug_boxes:
            print(f"Augmented boxes: {len(aug_boxes)}")
        
        return {
            'original_boxes': boxes,
            'merged_boxes': merged_boxes,
            'augmented_boxes': aug_boxes
        }