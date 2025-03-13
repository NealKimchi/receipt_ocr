# save this as test_dataset.py
import torch
from datasets import load_dataset
import json

def inspect_dataset():
    # Load the dataset
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v2", split='train')
    
    # Examine the first sample
    sample = dataset[0]
    
    # Print all available keys at the top level
    print(f"Top-level keys: {list(sample.keys())}")
    
    # Examine raw_data
    if 'raw_data' in sample:
        raw_data = sample['raw_data']
        print(f"\nRaw data type: {type(raw_data)}")
        
        if isinstance(raw_data, str):
            print(f"Raw data preview (string): {raw_data[:200]}...")
            try:
                parsed_raw = json.loads(raw_data)
                print(f"Raw data parsed as JSON: {type(parsed_raw)}")
                if isinstance(parsed_raw, dict):
                    print(f"Raw data JSON keys: {list(parsed_raw.keys())}")
            except:
                print("Failed to parse raw_data as JSON")
        elif isinstance(raw_data, dict):
            print(f"Raw data keys: {list(raw_data.keys())}")
            
            # Check first few values
            for key in list(raw_data.keys())[:3]:
                print(f"\nKey: {key}")
                print(f"Value type: {type(raw_data[key])}")
                print(f"Value preview: {str(raw_data[key])[:100]}...")
    
    # Examine parsed_data
    if 'parsed_data' in sample:
        parsed_data = sample['parsed_data']
        print(f"\nParsed data type: {type(parsed_data)}")
        
        if isinstance(parsed_data, dict):
            print(f"Parsed data keys: {list(parsed_data.keys())}")
            
            # Check first few values
            for key in list(parsed_data.keys())[:3]:
                print(f"\nKey: {key}")
                print(f"Value type: {type(parsed_data[key])}")
                print(f"Value preview: {str(parsed_data[key])[:100]}...")

if __name__ == "__main__":
    inspect_dataset()