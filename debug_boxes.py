# save as debug_boxes_simple.py
from datasets import load_dataset
import json

def debug_raw_data():
    # Load the dataset
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v2", split='train')
    
    # Try the first 5 samples
    for idx in range(5):
        sample = dataset[idx]
        print(f"\nSample {idx}")
        
        if 'raw_data' in sample:
            raw_data = sample['raw_data']
            print(f"Raw data type: {type(raw_data)}")
            
            # If raw_data is a string, try to parse it as JSON
            if isinstance(raw_data, str):
                try:
                    # Just print the raw string for manual inspection
                    print(f"Raw data preview: {raw_data[:200]}...")
                    
                    # Try to parse as JSON
                    data = json.loads(raw_data)
                    print(f"Successfully parsed as JSON")
                    
                    # Print the keys in the JSON
                    if isinstance(data, dict):
                        print(f"JSON keys: {list(data.keys())}")
                    
                    # Check if ocr_boxes exists and print its type
                    if 'ocr_boxes' in data:
                        print(f"ocr_boxes type: {type(data['ocr_boxes'])}")
                        print(f"ocr_boxes length: {len(data['ocr_boxes'])}")
                        
                        # Print the raw form of the first box for inspection
                        if len(data['ocr_boxes']) > 0:
                            print(f"First box raw: {data['ocr_boxes'][0]}")
                except json.JSONDecodeError as e:
                    print(f"Error parsing as JSON: {e}")
            else:
                print("Raw data is not a string")
        else:
            print("No raw_data field found")

if __name__ == "__main__":
    debug_raw_data()