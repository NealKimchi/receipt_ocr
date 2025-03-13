import os
import re
import numpy as np
import torch
import cv2

class CharsetMapper:
    """Class to handle character set mapping for text recognition"""
    def __init__(self, alphabet_path='data/alphabet.txt'):
        self.alphabet_path = alphabet_path
        self.charset, self.char_to_id, self.id_to_char = self._parse_alphabet()
        self.charset_size = len(self.charset)
        
    def _parse_alphabet(self):
        """Parse alphabet file to get character set and mappings"""
        charset = []
        # Read alphabet file
        with open(self.alphabet_path, 'r', encoding='utf-8') as f:
            for line in f:
                # Skip comments and empty lines
                if line.startswith('#') or line.strip() == '':
                    continue
                # Add each character in line to charset
                for char in line.strip():
                    if char not in charset:
                        charset.append(char)
        
        # Create char to id and id to char mappings (adding blank character for CTC)
        char_to_id = {char: i + 1 for i, char in enumerate(charset)}  # 0 reserved for blank (CTC)
        char_to_id[''] = 0  # Blank character for CTC
        id_to_char = {i + 1: char for i, char in enumerate(charset)}
        id_to_char[0] = ''  # Blank character for CTC
        
        return charset, char_to_id, id_to_char
    
    def encode(self, text):
        """Convert text to sequence of character IDs"""
        return [self.char_to_id.get(c, 0) for c in text]
    
    def decode(self, ids, remove_duplicates=True):
        """Convert sequence of character IDs to text"""
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().numpy()
            
        # Get characters
        text = ''.join([self.id_to_char.get(id, '') for id in ids if id > 0])
        
        # Remove consecutive duplicates if needed (for CTC decoding)
        if remove_duplicates:
            result = ""
            prev_char = None
            for char in text:
                if char != prev_char:
                    result += char
                    prev_char = char
            text = result
            
        return text
    
    def decode_predictions(self, predictions, raw=False):
        """
        Decode model predictions
        Args:
            predictions: Model output tensor of shape [batch_size, seq_len, num_classes]
            raw: Whether to return raw probabilities or decoded text
        Returns:
            List of decoded texts or raw probabilities
        """
        batch_size = predictions.shape[0]
        results = []
        
        for i in range(batch_size):
            # Get best indices
            best_indices = torch.argmax(predictions[i], dim=1) if len(predictions[i].shape) > 1 else predictions[i]
            
            if raw:
                results.append(best_indices)
            else:
                # Decode text
                text = self.decode(best_indices)
                results.append(text)
                
        return results


def preprocess_text(text):
    """Preprocess text for training"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters not in alphabet (optional)
    # text = re.sub(r'[^a-z0-9.,;:!?\'"\(\)\[\]\{\}\-_/\\&@#$%^*+=|~`<> ]', '', text)
    
    # Trim whitespace
    text = text.strip()
    
    return text


def crop_text_regions(image, boxes, padding=5):
    """
    Crop text regions from an image based on bounding boxes
    Args:
        image: Input image (numpy array)
        boxes: List of bounding boxes in format [x1, y1, x2, y2]
        padding: Padding to add around each box
    Returns:
        List of cropped regions
    """
    h, w = image.shape[:2]
    crops = []
    
    for box in boxes:
        # Unpack box coordinates
        x1, y1, x2, y2 = box
        
        # Add padding
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)
        
        # Ensure valid box
        if x2 <= x1 or y2 <= y1:
            continue
            
        # Crop region
        crop = image[y1:y2, x1:x2]
        crops.append(crop)
    
    return crops


def create_target_tensor(text, charset_mapper, max_length=32):
    """
    Create target tensor for training
    Args:
        text: Input text
        charset_mapper: CharsetMapper instance
        max_length: Maximum sequence length
    Returns:
        Encoded target tensor
    """
    # Encode text
    encoded = charset_mapper.encode(text)
    
    # Pad or truncate to max_length
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded = encoded + [0] * (max_length - len(encoded))
        
    return torch.tensor(encoded, dtype=torch.long)


def collate_text_recognition_batch(batch):
    """
    Collate function for text recognition batches
    
    This function ensures proper handling of variable-length text sequences
    and properly formats the batch for model input.
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    
    if not batch:
        return None
    
    # Separate data
    images = []
    texts = []
    raw_texts = []
    lengths = []
    image_ids = []
    
    for sample in batch:
        images.append(sample['image'])
        texts.append(sample['text'])
        raw_texts.append(sample['raw_text'])
        lengths.append(sample['length'])
        image_ids.append(sample['image_id'])
    
    images = torch.stack(images, dim=0)
    
    texts = torch.stack(texts, dim=0)
    
    # Convert lengths to tensor
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    return {
        'image': images,
        'text': texts,
        'raw_text': raw_texts,
        'length': lengths,
        'image_id': image_ids
    }