import json
import ast
import re

def extract_ocr_data(raw_data):
    """
    Extract OCR text and boxes from raw_data
    
    Args:
        raw_data: Raw OCR data from dataset
    
    Returns:
        dict: Dictionary with ocr_words and ocr_boxes if available
    """
    ocr_data = {'ocr_words': None, 'ocr_boxes': None}
    
    try:
        # Parse raw_data to JSON if it's a string
        if isinstance(raw_data, str):
            try:
                data = json.loads(raw_data)
            except json.JSONDecodeError:
                # Try to extract the ocr_words part directly
                match = re.search(r'"ocr_words":\s*(\[.*?\])', raw_data)
                if match:
                    words_str = match.group(1)
                    try:
                        ocr_data['ocr_words'] = ast.literal_eval(words_str)
                    except:
                        pass
                
                # Try to extract the ocr_boxes part directly
                match = re.search(r'"ocr_boxes":\s*(\[.*?\])', raw_data)
                if match:
                    boxes_str = match.group(1)
                    try:
                        ocr_data['ocr_boxes'] = ast.literal_eval(boxes_str)
                    except:
                        pass
                return ocr_data
        else:
            data = raw_data
        
        # Extract OCR words
        if 'ocr_words' in data:
            if isinstance(data['ocr_words'], str):
                try:
                    ocr_data['ocr_words'] = ast.literal_eval(data['ocr_words'])
                except:
                    ocr_data['ocr_words'] = [data['ocr_words']]
            else:
                ocr_data['ocr_words'] = data['ocr_words']
        
        # Extract OCR boxes
        if 'ocr_boxes' in data:
            if isinstance(data['ocr_boxes'], str):
                try:
                    ocr_data['ocr_boxes'] = ast.literal_eval(data['ocr_boxes'])
                except:
                    try:
                        ocr_data['ocr_boxes'] = json.loads(data['ocr_boxes'])
                    except:
                        ocr_data['ocr_boxes'] = data['ocr_boxes']
            else:
                ocr_data['ocr_boxes'] = data['ocr_boxes']
    
    except Exception as e:
        print(f"Error parsing OCR data: {e}")
    
    return ocr_data

def match_box_to_text(boxes, ocr_boxes, ocr_words):
    """
    Match detection boxes to OCR text based on IOU
    
    Args:
        boxes: Detection bounding boxes (normalized [x1, y1, x2, y2])
        ocr_boxes: OCR bounding boxes (pixel coordinates)
        ocr_words: OCR text for each box
    
    Returns:
        list: Matched text for each detection box
    """
    matched_text = []
    
    # If we don't have OCR data, return placeholder text
    if ocr_boxes is None or ocr_words is None:
        return [f"text_{i}" for i in range(len(boxes))]
    
    # Convert OCR boxes to normalized format if they're in pixel coordinates
    normalized_ocr_boxes = []
    for box_info in ocr_boxes:
        # Handle different OCR box formats
        try:
            if isinstance(box_info, list):
                # First item is box, second item is text and confidence
                box = box_info[0]
                # Convert to [x1, y1, x2, y2] format if needed
                if len(box) == 4:
                    normalized_ocr_boxes.append(box)
                elif len(box) == 2:  # [x, y] format
                    normalized_ocr_boxes.append([box[0][0], box[0][1], box[2][0], box[2][1]])
            elif isinstance(box_info, tuple) and len(box_info) == 2:
                # (box, (text, confidence)) format
                box = box_info[0]
                normalized_ocr_boxes.append(box)
        except:
            # Skip problematic boxes
            continue
    
    # For each detection box, find the best matching OCR box
    for box in boxes:
        best_match = None
        best_iou = 0
        
        for i, ocr_box in enumerate(normalized_ocr_boxes):
            # Calculate IOU
            iou = calculate_iou(box, ocr_box)
            
            if iou > best_iou:
                best_iou = iou
                if i < len(ocr_words):
                    best_match = ocr_words[i]
        
        if best_match:
            matched_text.append(best_match)
        else:
            matched_text.append(f"text_{len(matched_text)}")
    
    return matched_text

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1: First box [x1, y1, x2, y2]
        box2: Second box [x1, y1, x2, y2]
    
    Returns:
        float: IOU value
    """
    # Determine intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate area of intersection
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate area of both boxes
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Calculate IOU
    union = box1_area + box2_area - intersection
    if union <= 0:
        return 0.0
    
    return intersection / union