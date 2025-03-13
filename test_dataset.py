# save this as test_dataset.py
import torch
from datasets import load_dataset
import json

def test_dataset():
    # Load the dataset
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v2", split='train')
    
    # Look at the first 5 samples
    for i in range(5):
        sample = dataset[i]
        print(f"\nSample {i} keys: {list(sample.keys())}")
        
        # Check for OCR boxes
        if 'ocr_boxes' in sample:
            ocr_boxes = sample['ocr_boxes']
            print(f"OCR boxes type: {type(ocr_boxes)}")
            
            # If it's a string, it might be JSON
            if isinstance(ocr_boxes, str):
                print(f"OCR boxes string preview: {ocr_boxes[:100]}...")
                try:
                    parsed = json.loads(ocr_boxes)
                    print(f"Successfully parsed JSON. Result type: {type(parsed)}")
                    print(f"Parsed data length: {len(parsed) if isinstance(parsed, list) else 'not a list'}")
                except Exception as e:
                    print(f"Error parsing OCR boxes: {e}")
            elif isinstance(ocr_boxes, list):
                print(f"OCR boxes list length: {len(ocr_boxes)}")
                if len(ocr_boxes) > 0:
                    print(f"First box: {ocr_boxes[0]}")
        else:
            print("No 'ocr_boxes' field found")

if __name__ == "__main__":
    test_dataset()