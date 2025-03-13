import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLoss(nn.Module):
    """CTC Loss for sequence recognition"""
    def __init__(self, blank_idx=0, reduction='mean', zero_infinity=True):
        super(CTCLoss, self).__init__()
        self.criterion = nn.CTCLoss(
            blank=blank_idx,
            reduction=reduction,
            zero_infinity=zero_infinity
        )
        self.blank_idx = blank_idx
    
    def forward(self, outputs, targets, target_lengths=None):
        """
        Args:
            outputs: Model outputs containing:
                - log_probs: Log probabilities from model (T, B, C) for CTCLoss
            targets: Target sequences (B, max_length)
            target_lengths: Lengths of target sequences (B)
        Returns:
            loss: CTC loss value
        """
        log_probs = outputs['log_probs']
        predictions = outputs.get('predictions', None)
        
        # Get target sequences
        if isinstance(targets, dict):
            target_sequences = targets['text']
            target_lengths = targets['length']
        else:
            target_sequences = targets
            assert target_lengths is not None, "Target lengths must be provided"
        
        # Ensure target_sequences are in the correct format
        if isinstance(target_sequences[0], str):
            # Convert string targets to tensor - this would require a charset mapper
            # For this implementation, we assume targets are already numeric tensors
            raise ValueError("String targets not supported directly. Please convert to indices.")
        
        # Make sure target_sequences is a tensor
        if not isinstance(target_sequences, torch.Tensor):
            target_sequences = torch.tensor(target_sequences, dtype=torch.long, device=log_probs.device)
        
        # Calculate input lengths (needed for CTC)
        # For log_probs of shape (T, B, C), we need input_lengths of shape (B)
        input_lengths = torch.full(
            size=(log_probs.size(1),),
            fill_value=log_probs.size(0),
            dtype=torch.long,
            device=log_probs.device
        )
        
        # Compute CTC loss
        loss = self.criterion(
            log_probs, 
            target_sequences, 
            input_lengths, 
            target_lengths
        )
        
        return loss


class AttentionLoss(nn.Module):
    """Attention-based cross-entropy loss for sequence prediction"""
    def __init__(self, ignore_index=0, label_smoothing=0.0):
        super(AttentionLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(self, outputs, targets, target_lengths=None):
        """
        Args:
            outputs: Model outputs containing:
                - logits: Class logits from model (B, T, C)
            targets: Target sequences (B, T)
            target_lengths: Lengths of target sequences (not used here)
        Returns:
            loss: Cross-entropy loss value
        """
        logits = outputs['logits']
        
        # Get target sequences
        if isinstance(targets, dict):
            target_sequences = targets['text']
        else:
            target_sequences = targets
        
        # Ensure targets are in tensor format
        if not isinstance(target_sequences, torch.Tensor):
            target_sequences = torch.tensor(target_sequences, dtype=torch.long, device=logits.device)
        
        # Reshape for cross entropy:
        # (B, T, C) -> (B*T, C) for logits
        # (B, T) -> (B*T) for targets
        batch_size, seq_len, num_classes = logits.size()
        logits_flat = logits.reshape(-1, num_classes)
        targets_flat = target_sequences.reshape(-1)
        
        # Compute loss
        loss = self.criterion(logits_flat, targets_flat)
        
        return loss


class CombinedLoss(nn.Module):
    """Combined CTC and Attention loss"""
    def __init__(self, ctc_weight=0.5, blank_idx=0, ignore_idx=0, label_smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.ctc_loss = CTCLoss(blank_idx=blank_idx)
        self.attention_loss = AttentionLoss(ignore_index=ignore_idx, label_smoothing=label_smoothing)
        self.ctc_weight = ctc_weight
    
    def forward(self, outputs, targets, target_lengths=None):
        """
        Args:
            outputs: Model outputs containing both CTC and attention outputs
            targets: Target sequences
            target_lengths: Lengths of target sequences
        Returns:
            loss: Combined loss value
        """
        ctc_loss = self.ctc_loss(outputs, targets, target_lengths)
        att_loss = self.attention_loss(outputs, targets, target_lengths)
        
        # Combine losses with weighting
        loss = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * att_loss
        
        # Return individual losses as well
        return {
            'loss': loss,
            'ctc_loss': ctc_loss,
            'attention_loss': att_loss
        }


def get_loss_function(config):
    """Create the loss function based on configuration"""
    loss_type = config['loss']['type']
    
    if loss_type == 'ctc':
        return CTCLoss(
            blank_idx=config['loss']['blank_index'],
            zero_infinity=True
        )
    elif loss_type == 'attention':
        return AttentionLoss(
            ignore_index=config['loss']['blank_index'],
            label_smoothing=config['loss'].get('label_smoothing', 0.0)
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            ctc_weight=config['loss']['ctc_weight'],
            blank_idx=config['loss']['blank_index'],
            ignore_idx=config['loss']['blank_index'],
            label_smoothing=config['loss'].get('label_smoothing', 0.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")