import torch
import re
import numpy as np
from difflib import SequenceMatcher

def preprocess_text(text, case_sensitive=False, ignore_punctuation=True):
    """Preprocess text for metrics calculation"""
    # Handle case sensitivity
    if not case_sensitive:
        text = text.lower()
    
    # Remove punctuation if specified
    if ignore_punctuation:
        text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def calculate_cer(pred_text, target_text, case_sensitive=False, ignore_punctuation=True):
    """
    Calculate Character Error Rate (CER)
    CER = (S + D + I) / N
    where:
        S = number of substitutions
        D = number of deletions
        I = number of insertions
        N = number of characters in the target text
    """
    # Preprocess texts
    pred_text = preprocess_text(pred_text, case_sensitive, ignore_punctuation)
    target_text = preprocess_text(target_text, case_sensitive, ignore_punctuation)
    
    # If both strings are empty, return 0 error rate
    if len(target_text) == 0:
        return 0.0 if len(pred_text) == 0 else 1.0
    
    # Calculate edit distance (Levenshtein distance)
    m = len(pred_text)
    n = len(target_text)
    
    # Create distance matrix
    distance = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j
    
    # Fill distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_text[i - 1] == target_text[j - 1]:
                cost = 0
            else:
                cost = 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,      # deletion
                distance[i][j - 1] + 1,      # insertion
                distance[i - 1][j - 1] + cost # substitution
            )
    
    # Return CER
    return distance[m][n] / n

def calculate_wer(pred_text, target_text, case_sensitive=False, ignore_punctuation=True):
    """
    Calculate Word Error Rate (WER)
    WER = (S + D + I) / N
    where:
        S = number of substituted words
        D = number of deleted words
        I = number of inserted words
        N = number of words in the target text
    """
    # Preprocess texts
    pred_text = preprocess_text(pred_text, case_sensitive, ignore_punctuation)
    target_text = preprocess_text(target_text, case_sensitive, ignore_punctuation)
    
    # Convert to word arrays
    pred_words = pred_text.split()
    target_words = target_text.split()
    
    # If target is empty, return 0 error rate if prediction is also empty, 1.0 otherwise
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    # Calculate edit distance at word level
    m = len(pred_words)
    n = len(target_words)
    
    # Create distance matrix
    distance = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Initialize first row and column
    for i in range(m + 1):
        distance[i][0] = i
    for j in range(n + 1):
        distance[0][j] = j
    
    # Fill distance matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_words[i - 1] == target_words[j - 1]:
                cost = 0
            else:
                cost = 1
            distance[i][j] = min(
                distance[i - 1][j] + 1,       # deletion
                distance[i][j - 1] + 1,       # insertion
                distance[i - 1][j - 1] + cost # substitution
            )
    
    # Return WER
    return distance[m][n] / n

def calculate_sequence_accuracy(predictions, targets):
    """
    Calculate exact sequence match accuracy
    Args:
        predictions: List of predicted texts
        targets: List of target texts
    Returns:
        accuracy: Percentage of exact matches
    """
    correct = 0
    total = len(predictions)
    
    for pred, target in zip(predictions, targets):
        if pred == target:
            correct += 1
    
    return correct / total if total > 0 else 0.0

def calculate_text_metrics(predictions, targets, case_sensitive=False, ignore_punctuation=True):
    """
    Calculate various text recognition metrics
    Args:
        predictions: List of predicted texts
        targets: List of target texts
        case_sensitive: Whether to be sensitive to case
        ignore_punctuation: Whether to ignore punctuation
    Returns:
        metrics: Dictionary of metrics
    """
    num_samples = len(predictions)
    total_cer = 0.0
    total_wer = 0.0
    correct_sequences = 0
    
    for pred, target in zip(predictions, targets):
        # Calculate CER
        cer = calculate_cer(pred, target, case_sensitive, ignore_punctuation)
        total_cer += cer
        
        # Calculate WER
        wer = calculate_wer(pred, target, case_sensitive, ignore_punctuation)
        total_wer += wer
        
        # Check if exact match (after preprocessing)
        pred_processed = preprocess_text(pred, case_sensitive, ignore_punctuation)
        target_processed = preprocess_text(target, case_sensitive, ignore_punctuation)
        if pred_processed == target_processed:
            correct_sequences += 1
    
    # Calculate averages
    avg_cer = total_cer / num_samples if num_samples > 0 else 0.0
    avg_wer = total_wer / num_samples if num_samples > 0 else 0.0
    accuracy = correct_sequences / num_samples if num_samples > 0 else 0.0
    
    # Create metrics dictionary
    metrics = {
        'cer': avg_cer,
        'wer': avg_wer,
        'accuracy': accuracy,
        'num_samples': num_samples
    }
    
    return metrics

def print_metrics(metrics):
    """Print formatted metrics"""
    print(f"Metrics over {metrics['num_samples']} samples:")
    print(f"  Character Error Rate (CER): {metrics['cer']:.4f}")
    print(f"  Word Error Rate (WER): {metrics['wer']:.4f}")
    print(f"  Sequence Accuracy: {metrics['accuracy']:.4f}")