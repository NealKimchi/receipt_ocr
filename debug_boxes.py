# save as debug_boxes.py
from datasets import load_dataset
import json

def debug_raw_data():
    # Load the dataset
    dataset = load_dataset("mychen76/invoices-and-receipts_ocr_v2", split='train')
    
    # Try the first 10 samples
    for idx in range(10):
        sample = dataset[idx]
        print(f"\nSample {idx}")
        
        if 'raw_data' in sample:
            # Check if raw_data is a string
            if isinstance(sample['raw_data'], str):
                # Try to parse as JSON
                try:
                    data = json.loads(sample['raw_data'])
                    print(f"Successfully parsed raw_data as JSON")
                    
                    # Check if ocr_boxes exists
                    if 'ocr_boxes' in data:
                        boxes = data['ocr_boxes']
                        print(f"Found ocr_boxes, length: {len(boxes)}")
                        
                        # Print the first box directly
                        if len(boxes) > 0:
                            print(f"First box raw: {boxes[0]}")
                            
                            # Try to access expected elements
                            try:
                                polygon = boxes[0][0]
                                text_conf = boxes[0][1]
                                print(f"Polygon: {polygon}")
                                print(f"Text and confidence: {text_conf}")
                            except (IndexError, TypeError) as e:
                                print(f"Error accessing box elements: {e}")
                    else:
                        print("No 'ocr_boxes' found in parsed raw_data")
                except json.JSONDecodeError as e:
                    print(f"Error parsing raw_data as JSON: {e}")
            else:
                print(f"raw_data is not a string, type: {type(sample['raw_data'])}")
        else:
            print("No 'raw_data' field found")

if __name__ == "__main__":
    debug_raw_data()