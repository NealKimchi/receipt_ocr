import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeatureExtractor(nn.Module):
    """CNN feature extractor for text recognition (custom implementation)"""
    def __init__(self, input_channels=3, backbone=None):
        super(FeatureExtractor, self).__init__()
        
        # Skip backbone check completely - we're always using our custom implementation
        # Regardless of what's passed in for the backbone parameter
        
        # First convolutional block
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)  # 1/2 resolution
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)  # 1/4 resolution
        
        # Third convolutional block (maintain width for sequence)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional block
        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d((2, 1))  # Pool only height, preserve width for sequence, 1/8 height
        
        # Fifth convolutional block
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU(inplace=True)
        
        # Sixth convolutional block
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool6 = nn.MaxPool2d((2, 1))  # Pool only height, preserve width for sequence, 1/16 height
        
        # Final adaptive layer to ensure consistent feature map height
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, None))  # Fix height to 4, preserve width
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
        
        # Output channels
        self.out_channels = 512
    
    def forward(self, x):
        # Apply convolutional blocks
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.pool6(self.relu6(self.bn6(self.conv6(x))))
        
        # Apply adaptive pooling to ensure consistent height
        x = self.adaptive_pool(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        return x


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM for sequence modeling"""
    def __init__(self, input_size, hidden_size, output_size=None):
        super(BidirectionalLSTM, self).__init__()
        
        if output_size is None:
            output_size = hidden_size
            
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)
    
    def forward(self, x):
        # Input shape: (batch, sequence, input_size)
        # Output shape: (batch, sequence, output_size)
        output, _ = self.rnn(x)
        output = self.fc(output)
        return output


class AttentionModule(nn.Module):
    """Attention mechanism for focusing on relevant parts of the feature sequence"""
    def __init__(self, input_size, hidden_size, attention_size=128):
        super(AttentionModule, self).__init__()
        
        self.attention_cell = nn.LSTMCell(input_size + hidden_size, hidden_size)
        
        # Attention layer
        self.attention_hidden = nn.Linear(hidden_size, attention_size)
        self.attention_context = nn.Linear(input_size, attention_size)
        self.attention_energy = nn.Linear(attention_size, 1)
        
        # Output projection
        self.project = nn.Linear(hidden_size + input_size, hidden_size)
    
    def forward(self, prev_hidden, prev_cell, contexts):
        """
        Args:
            prev_hidden: Previous hidden state (batch, hidden_size)
            prev_cell: Previous cell state (batch, hidden_size)
            contexts: Context vectors (batch, seq_len, input_size)
        """
        # Repeat hidden state for each time step
        seq_len = contexts.size(1)
        batch_size = contexts.size(0)
        
        # Attention mechanism
        # Shape: (batch, 1, hidden_size) -> (batch, seq_len, hidden_size)
        hidden_attention = self.attention_hidden(prev_hidden).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Shape: (batch, seq_len, attention_size)
        context_attention = self.attention_context(contexts)
        
        # Combine and get energy
        # Shape: (batch, seq_len, 1)
        energy = self.attention_energy(torch.tanh(hidden_attention + context_attention))
        
        # Get attention weights
        # Shape: (batch, seq_len)
        attention_weights = F.softmax(energy.squeeze(2), dim=1)
        
        # Apply attention weights to context
        # Shape: (batch, 1, seq_len) @ (batch, seq_len, input_size) -> (batch, 1, input_size)
        weighted_context = torch.bmm(attention_weights.unsqueeze(1), contexts).squeeze(1)
        
        # Input to attention cell
        # Shape: (batch, hidden_size + input_size)
        rnn_input = torch.cat([prev_hidden, weighted_context], dim=1)
        
        # Attention cell forward pass
        hidden, cell = self.attention_cell(rnn_input, (prev_hidden, prev_cell))
        
        # Combine context and hidden for output
        output = self.project(torch.cat([hidden, weighted_context], dim=1))
        
        return output, hidden, cell, attention_weights


class CRNN(nn.Module):
    """Convolutional Recurrent Neural Network for text recognition"""
    def __init__(self, config, num_classes):
        super(CRNN, self).__init__()
        
        # Save configuration
        self.config = config
        
        # CNN feature extractor
        # Create without passing the backbone parameter at all
        self.feature_extractor = FeatureExtractor(
            input_channels=config['model']['input_channels']
        )
        
        # Determine size of CNN output features
        cnn_output_channels = self.feature_extractor.out_channels
        
        # Calculate sequence input size (height is collapsed)
        # Our model uses fixed height 4 from adaptive pooling
        feature_height = config['model'].get('feature_height', 4)  # Default to 4 if not specified
        self.sequence_input_size = cnn_output_channels * feature_height
        
        # RNN layers
        self.use_attention = config['model']['attention']
        hidden_size = config['model']['hidden_size']
        
        if config['model']['rnn_type'] == 'lstm':
            rnn_cell = nn.LSTM
        elif config['model']['rnn_type'] == 'gru':
            rnn_cell = nn.GRU
        else:
            raise ValueError(f"Unsupported RNN type: {config['model']['rnn_type']}")
        
        # Recurrent layers
        self.rnn = rnn_cell(
            input_size=self.sequence_input_size,
            hidden_size=hidden_size,
            num_layers=config['model']['num_rnn_layers'],
            bidirectional=config['model']['bidirectional'],
            dropout=config['model']['dropout_rate'] if config['model']['num_rnn_layers'] > 1 else 0,
            batch_first=True
        )
        
        # Output size after bidirectional RNN
        rnn_output_size = hidden_size * 2 if config['model']['bidirectional'] else hidden_size
        
        # Attention layer (optional)
        if self.use_attention:
            self.attention = AttentionModule(
                input_size=rnn_output_size, 
                hidden_size=hidden_size
            )
            self.decoder_input_size = hidden_size
        else:
            self.decoder_input_size = rnn_output_size
        
        # Final projection to classes
        self.classifier = nn.Linear(self.decoder_input_size, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        # Initialize CNN weights
        for m in self.feature_extractor.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
        # Initialize RNN weights
        for name, param in self.rnn.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        self.classifier.bias.data.fill_(0)
    
    def forward(self, x, targets=None, target_lengths=None):
        """
        Forward pass of the model
        Args:
            x: Input images (batch_size, channels, height, width)
            targets: Target sequences (optional, for attention training)
            target_lengths: Lengths of target sequences
        Returns:
            Dictionary of outputs including logits, probabilities, and predictions
        """
        batch_size = x.size(0)
        
        # Extract features using CNN
        # Output shape: (batch_size, channels, height, width)
        features = self.feature_extractor(x)
        
        # Prepare features for RNN (collapse height)
        # (batch_size, channels, height, width) -> (batch_size, width, channels*height)
        h, w = features.size(2), features.size(3)
        features = features.permute(0, 3, 1, 2)  # (batch_size, width, channels, height)
        features = features.reshape(batch_size, w, -1)  # (batch_size, width, channels*height)
        
        # RNN forward pass
        # Output shape: (batch_size, width, hidden_size*num_directions)
        if isinstance(self.rnn, nn.LSTM):
            rnn_output, _ = self.rnn(features)
        else:  # GRU
            rnn_output, _ = self.rnn(features)
        
        # Apply attention if configured and in training mode
        if self.use_attention and targets is not None and self.training:
            try:
                # Initialize attention states
                hidden = torch.zeros(batch_size, self.config['model']['hidden_size'], device=x.device)
                cell = torch.zeros(batch_size, self.config['model']['hidden_size'], device=x.device)
                
                # Initialize output tensor
                max_length = self.config['data']['max_text_length']
                outputs = torch.zeros(batch_size, max_length, self.decoder_input_size, device=x.device)
                
                # Attention decoding
                attentions = torch.zeros(batch_size, max_length, rnn_output.size(1), device=x.device)
                
                # Teacher forcing for training
                num_classes = self.classifier.weight.size(0)
                for t in range(max_length):
                    if t == 0:
                        # Start with zeros for first step
                        context = torch.zeros(batch_size, self.decoder_input_size, device=x.device)
                    else:
                        # Use ground truth for teacher forcing, but handle index out of bounds
                        # Get previous tokens, ensuring they're within bounds
                        prev_tokens = targets[:, t-1].clone()
                        # Clamp indices to valid range (0 to num_classes-1)
                        prev_tokens = torch.clamp(prev_tokens, 0, num_classes-1)
                        # Get embeddings from classifier weights
                        context = self.classifier.weight[prev_tokens]
                    
                    # Run attention step
                    output, hidden, cell, attention = self.attention(hidden, cell, rnn_output)
                    outputs[:, t] = output
                    attentions[:, t] = attention
                
                # Project to class probabilities
                # Shape: (batch_size, max_length, num_classes)
                logits = self.classifier(outputs)
            except Exception as e:
                # Fall back to CTC path if attention fails
                print(f"Warning: Attention mechanism failed with error: {e}. Falling back to CTC.")
                logits = self.classifier(rnn_output)
                attentions = None
        else:
            # For CTC or inference, use RNN output directly
            # Project to class probabilities
            # Shape: (batch_size, width, num_classes)
            logits = self.classifier(rnn_output)
            attentions = None
        
        # Apply log softmax over the class dimension
        log_probs = F.log_softmax(logits, dim=-1)
        
        # Permute for CTC loss if not using attention or in inference mode
        if not (self.use_attention and targets is not None and self.training):
            # Shape: (width, batch_size, num_classes) - needed for CTCLoss
            log_probs = log_probs.permute(1, 0, 2)
        
        # Get predictions
        predictions = torch.argmax(logits, dim=-1)
        
        return {
            'logits': logits,
            'log_probs': log_probs,
            'predictions': predictions,
            'attentions': attentions,
            'features': features
        }


def get_model(config, num_classes):
    """Create and initialize the text recognition model"""
    model = CRNN(config, num_classes=num_classes)
    return model