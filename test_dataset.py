# save this as test_dataset.py
import torch
from datasets import load_dataset
import json

# Modified test script
def test_dataset():
    # Load the dataset
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v2", split='train')
    
    # Look at the first sample in more detail
    sample = dataset[0]
    
    # Check if raw_data contains OCR information
    if 'raw_data' in sample:
        print("\nSample raw_data keys:")
        if isinstance(sample['raw_data'], dict):
            print(list(sample['raw_data'].keys()))
            
            # Check if ocr_boxes is in raw_data
            if 'ocr_boxes' in sample['raw_data']:
                print("\nFound ocr_boxes in raw_data!")
                print(f"Type: {type(sample['raw_data']['ocr_boxes'])}")
                print(f"Preview: {str(sample['raw_data']['ocr_boxes'])[:100]}...")
    
    # Check if parsed_data contains OCR information
    if 'parsed_data' in sample:
        print("\nSample parsed_data keys:")
        if isinstance(sample['parsed_data'], dict):
            print(list(sample['parsed_data'].keys()))
            
            # Check if any field might contain box information
            for key in sample['parsed_data'].keys():
                print(f"\nKey: {key}")
                print(f"Type: {type(sample['parsed_data'][key])}")
                print(f"Preview: {str(sample['parsed_data'][key])[:100]}...")