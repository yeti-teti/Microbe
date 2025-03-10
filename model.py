import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class LayerNorm(nn.Module):
    """Layer normalization module"""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionalEncoding(nn.Module):
    """Implement the positional encoding function"""
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].requires_grad_(False)
        return self.dropout(x)

class PeakEncoder(nn.Module):
    """Simple encoder for MS2 spectrum peaks"""
    def __init__(self, max_peaks, d_model):
        super(PeakEncoder, self).__init__()
        
        # Project peak features to model dimension
        self.projection = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Position encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout=0.1, max_len=max_peaks)
        
        # Transformer-style blocks using PyTorch's TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, peaks_mz, peaks_intensity):
        # Combine m/z and intensity values [batch, max_peaks, 2]
        peaks = torch.stack([peaks_mz, peaks_intensity], dim=2)
        
        # Project to model dimension [batch, max_peaks, d_model]
        x = self.projection(peaks)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask (True = padding position to be masked)
        padding_mask = (peaks_mz == 0)
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        
        return x

class PrecursorEncoder(nn.Module):
    """Encodes precursor information (m/z and charge)"""
    def __init__(self, d_model):
        super(PrecursorEncoder, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model)
        )
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, precursor):
        # precursor: [batch, 2] with [mz, charge]
        x = self.projection(precursor)  # [batch, d_model]
        x = self.norm(x)
        return x.unsqueeze(1)  # [batch, 1, d_model]

class Embeddings(nn.Module):
    """Embedding layer with scaling"""
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

def generate_square_subsequent_mask(sz, device):
    """Generate a square mask for the sequence. The masked positions are filled with float('-inf').
    Unmasked positions are filled with float(0.0).
    """
    mask = (torch.triu(torch.ones(sz, sz, device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class SequenceGenerator(nn.Module):
    """Define projection to vocabulary for sequence prediction"""
    def __init__(self, d_model, vocab_size):
        super(SequenceGenerator, self).__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, temperature=1.0):
        """
        Forward pass with optional temperature scaling
        
        Args:
            x: Input tensor
            temperature: Temperature for softmax (lower = more peaked distribution)
            
        Returns:
            Log probabilities for each vocabulary item
        """
        logits = self.proj(x)
        return F.log_softmax(logits / temperature, dim=-1)

class PTMLocalPredictor(nn.Module):
    """Predicts local PTM presence and mass offset"""
    def __init__(self, d_model):
        super(PTMLocalPredictor, self).__init__()
        self.ptm_presence = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1),
            nn.Sigmoid()
        )
        
        self.ptm_offset = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: Transformer decoder output [batch, seq_len, d_model]
        
        Returns:
            ptm_presence: Probability of PTM at each position [batch, seq_len]
            ptm_offset: Predicted mass shift at each position [batch, seq_len]
        """
        ptm_presence = self.ptm_presence(x).squeeze(-1)
        ptm_offset = self.ptm_offset(x).squeeze(-1)
        
        return ptm_presence, ptm_offset

class PTMGlobalPredictor(nn.Module):
    """Predicts global PTM presence and total mass offset"""
    def __init__(self, d_model):
        super(PTMGlobalPredictor, self).__init__()
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        # Global prediction heads
        self.ptm_presence = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1),
            nn.Sigmoid()
        )
        
        self.ptm_offset = nn.Sequential(
            nn.Linear(d_model, d_model//2),
            nn.ReLU(),
            nn.Linear(d_model//2, 1)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: Decoder output [batch, seq_len, d_model]
            mask: Sequence mask [batch, seq_len]
            
        Returns:
            ptm_presence: Global PTM presence [batch]
            ptm_offset: Global PTM offset [batch]
        """
        # Compute attention weights
        attn_weights = self.attention(x)  # [batch, seq_len, 1]
        
        if mask is not None:
            # Apply mask to attention weights
            mask = mask.float().unsqueeze(-1)
            attn_weights = attn_weights * mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
        
        # Weighted pooling 
        global_repr = (x * attn_weights).sum(dim=1)  # [batch, d_model]
        
        # Predict global PTM presence and offset
        ptm_presence = self.ptm_presence(global_repr).squeeze(-1)
        ptm_offset = self.ptm_offset(global_repr).squeeze(-1)
        
        return ptm_presence, ptm_offset

class Spec2PepWithPTM(nn.Module):
    """
    Model for spectrum to peptide sequence prediction with PTM identification.
    Uses PyTorch's transformer components for simplicity and stability.
    """
    def __init__(
        self, 
        vocab_size,
        max_peaks=200,
        max_seq_len=50,
        d_model=512, 
        d_ff=2048, 
        n_heads=8, 
        n_layers=6, 
        dropout=0.1
    ):
        super(Spec2PepWithPTM, self).__init__()
        
        # Save configuration
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Encoders
        self.peak_encoder = PeakEncoder(max_peaks, d_model)
        self.precursor_encoder = PrecursorEncoder(d_model)
        
        # Sequence embedding
        self.token_embedding = Embeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer decoder using PyTorch's transformer components
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output heads
        self.sequence_generator = SequenceGenerator(d_model, vocab_size)
        self.ptm_local_predictor = PTMLocalPredictor(d_model)
        self.ptm_global_predictor = PTMGlobalPredictor(d_model)
        
        # Initialize parameters with custom initialization
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters with custom values to prevent mode collapse"""
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Different initialization based on parameter type
                if 'proj' in name or 'generator' in name:
                    # Output projections - more aggressive initialization
                    nn.init.xavier_uniform_(p, gain=1.4)
                elif 'embedding' in name or 'token_embedding' in name:
                    # Embeddings - smaller values
                    nn.init.normal_(p, mean=0.0, std=0.1)
                else:
                    # Default initialization
                    nn.init.xavier_uniform_(p, gain=1.0)
    
    def forward(self, features, teacher_forcing_ratio=1.0, temperature=1.0):
        """
        Forward pass through the model with optional teacher forcing and temperature scaling.
        
        Args:
            features: Dictionary containing:
                - 'peaks': Tuple of (mz_values, intensity_values)
                - 'precursor': Precursor info tensor [batch, 2]
                - 'input_sequence': Input sequence token IDs [batch, seq_len]
                - 'attention_mask': Boolean mask for sequence (True = use, False = ignore)
            teacher_forcing_ratio: Probability of using teacher forcing (1.0 = always use ground truth)
            temperature: Temperature parameter for softmax (lower = more peaked distribution)
        
        Returns:
            Dictionary of model outputs
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Extract inputs and move to the correct device
        peaks = features['peaks']
        peaks_mz, peaks_intensity = peaks[0].to(device), peaks[1].to(device)
        precursor = features['precursor'].to(device)
        seq_tokens = features['input_sequence'].to(device)
        
        # Process peaks
        peak_encoding = self.peak_encoder(peaks_mz, peaks_intensity)  # [batch, max_peaks, d_model]
        
        # Process precursor
        precursor_encoding = self.precursor_encoder(precursor)  # [batch, 1, d_model]
        
        # Combine encoder outputs
        memory = torch.cat([precursor_encoding, peak_encoding], dim=1)  # [batch, 1+max_peaks, d_model]
        
        # Create memory padding mask (True = padding position to be masked)
        memory_key_padding_mask = torch.zeros(
            (memory.size(0), memory.size(1)), 
            device=device,
            dtype=torch.bool
        )
        # Set the peaks part of the mask using the mz values
        memory_key_padding_mask[:, 1:] = (peaks_mz == 0)
        
        # Determine if we're using teacher forcing or autoregressive decoding
        if self.training and torch.rand(1).item() <= teacher_forcing_ratio:
            # Teacher forcing mode - standard transformer decoding
            
            # Embed sequence tokens
            seq_embedded = self.token_embedding(seq_tokens)  # [batch, seq_len, d_model]
            seq_embedded = self.pos_encoding(seq_embedded)
            
            # Create sequence padding mask (True = padding position to be masked)
            if 'attention_mask' in features:
                seq_padding_mask = ~features['attention_mask'].to(device)  # Convert from "attend" to "mask"
            else:
                seq_padding_mask = None
            
            # Create causal mask for autoregressive decoding
            seq_len = seq_tokens.size(1)
            tgt_mask = generate_square_subsequent_mask(seq_len, device)
            
            # Apply transformer decoder
            decoder_output = self.transformer_decoder(
                seq_embedded, 
                memory,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_key_padding_mask=seq_padding_mask
            )
            
            # Generate sequence predictions
            sequence_probs = self.sequence_generator(decoder_output, temperature)
            
        else:
            # Autoregressive mode - predict one token at a time
            batch_size = seq_tokens.size(0)
            seq_len = seq_tokens.size(1)
            
            # Initialize output tensors
            sequence_probs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
            decoder_outputs = torch.zeros(batch_size, seq_len, self.d_model).to(device)
            
            # Start with just the first token (usually <sos>)
            current_input = seq_tokens[:, 0].unsqueeze(1)  # [batch, 1]
            
            # Generate sequence autoregressively
            for t in range(seq_len):
                # Embed current sequence
                current_embedded = self.token_embedding(current_input)  # [batch, t+1, d_model]
                current_embedded = self.pos_encoding(current_embedded)
                
                # Create causal mask for current length
                curr_len = current_input.size(1)
                curr_mask = generate_square_subsequent_mask(curr_len, device)
                
                # Decode current sequence
                current_output = self.transformer_decoder(
                    current_embedded,
                    memory,
                    tgt_mask=curr_mask,
                    memory_key_padding_mask=memory_key_padding_mask
                )
                
                # Store decoder output for latest position
                if t < seq_len:
                    decoder_outputs[:, t] = current_output[:, -1]
                
                # Get prediction probabilities for the latest position
                current_probs = self.sequence_generator(current_output[:, -1].unsqueeze(1), temperature)
                
                # Store predictions
                if t < seq_len:
                    sequence_probs[:, t] = current_probs.squeeze(1)
                
                # Prepare input for next iteration
                if t + 1 < seq_len:
                    # During training, sometimes use the ground truth (teacher forcing)
                    if self.training and torch.rand(1).item() <= teacher_forcing_ratio:
                        next_token = seq_tokens[:, t+1].unsqueeze(1)
                    else:
                        # Otherwise use model prediction (with optional sampling)
                        if temperature > 0:
                            # Sample from distribution
                            probs = torch.exp(current_probs.squeeze(1))
                            next_token = torch.multinomial(probs, 1)
                        else:
                            # Greedy decoding
                            next_token = current_probs.argmax(dim=-1).unsqueeze(1)
                    
                    # Append next token to current input
                    current_input = torch.cat([current_input, next_token], dim=1)
            
            # Replace decoder_output with our accumulated outputs
            decoder_output = decoder_outputs
        
        # Generate local PTM predictions
        ptm_local_presence, ptm_local_offset = self.ptm_local_predictor(decoder_output)
        
        # Generate global PTM predictions
        attention_mask = features.get('attention_mask', None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        ptm_global_presence, ptm_global_offset = self.ptm_global_predictor(
            decoder_output, 
            mask=attention_mask
        )
        
        return {
            'sequence_probs': sequence_probs,
            'ptm_local_presence': ptm_local_presence,
            'ptm_local_offset': ptm_local_offset,
            'ptm_global_presence': ptm_global_presence,
            'ptm_global_offset': ptm_global_offset
        }
    
    def predict_sequence(self, features, max_length=None, temperature=0.7, top_k=0, top_p=0.9):
        """
        Generate a sequence prediction with various decoding strategies.
        
        Args:
            features: Input features dictionary
            max_length: Maximum sequence length (default: self.max_seq_len)
            temperature: Temperature for sampling (lower = more peaked, 0 = greedy)
            top_k: Keep only top k tokens with highest probability (0 = all)
            top_p: Keep top tokens with cumulative probability >= top_p (1.0 = all)
            
        Returns:
            Dictionary with predictions
        """
        if max_length is None:
            max_length = self.max_seq_len
            
        # Get device
        device = next(self.parameters()).device
        
        # Extract inputs and move to device
        peaks = features['peaks']
        peaks_mz, peaks_intensity = peaks[0].to(device), peaks[1].to(device)
        precursor = features['precursor'].to(device)
        
        batch_size = peaks_mz.size(0)
        
        # Process peaks and precursor
        peak_encoding = self.peak_encoder(peaks_mz, peaks_intensity)
        precursor_encoding = self.precursor_encoder(precursor)
        memory = torch.cat([precursor_encoding, peak_encoding], dim=1)
        
        # Create memory padding mask
        memory_key_padding_mask = torch.zeros(
            (memory.size(0), memory.size(1)), 
            device=device,
            dtype=torch.bool
        )
        memory_key_padding_mask[:, 1:] = (peaks_mz == 0)
        
        # Start with SOS token (typically index 1 but may vary)
        # If 'input_sequence' is provided, use its first token, otherwise default to index 1
        if 'input_sequence' in features:
            current_input = features['input_sequence'][:, 0].unsqueeze(1).to(device)
        else:
            # Assuming SOS token index is 1
            current_input = torch.ones(batch_size, 1, dtype=torch.long, device=device)
        
        # Initialize containers for output
        all_tokens = [current_input]
        all_scores = []
        
        # Autoregressive generation
        for step in range(max_length - 1):
            # Embed current sequence
            current_embedded = self.token_embedding(current_input)
            current_embedded = self.pos_encoding(current_embedded)
            
            # Create mask
            curr_len = current_input.size(1)
            curr_mask = generate_square_subsequent_mask(curr_len, device)
            
            # Decode
            decoder_output = self.transformer_decoder(
                current_embedded,
                memory,
                tgt_mask=curr_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
            
            # Get prediction for next token
            logits = self.sequence_generator.proj(decoder_output[:, -1, :])
            
            # Apply temperature
            if temperature > 0:
                logits = logits / temperature
            
            # Apply softmax to get probabilities
            scores = F.softmax(logits, dim=-1)
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
                scores[indices_to_remove] = 0
                
            # Apply top-p (nucleus) filtering
            if 0 < top_p < 1.0:
                sorted_scores, sorted_indices = torch.sort(scores, descending=True)
                cumulative_probs = torch.cumsum(sorted_scores, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep first probs above threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Create a mask for indices to remove
                indices_to_remove = torch.zeros_like(scores, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                scores[indices_to_remove] = 0
            
            # Renormalize probabilities if needed
            if top_k > 0 or 0 < top_p < 1.0:
                scores = scores / scores.sum(dim=-1, keepdim=True)
            
            # Sample from the filtered distribution
            if temperature > 0:
                next_token = torch.multinomial(scores, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(scores, dim=-1).unsqueeze(-1)
            
            # Append to outputs
            all_tokens.append(next_token)
            all_scores.append(scores)
            
            # Update input for next iteration
            current_input = torch.cat([current_input, next_token], dim=1)
        
        # Process all predictions
        all_tokens = torch.cat(all_tokens, dim=1)  # [batch, max_length]
        
        # Generate a full forward pass with the final sequence to get PTM predictions
        features_updated = {
            'peaks': features['peaks'],
            'precursor': features['precursor'],
            'input_sequence': all_tokens,
            'attention_mask': torch.ones_like(all_tokens, dtype=torch.bool)
        }
        
        # Run inference with the predicted sequence
        with torch.no_grad():
            outputs = self.forward(features_updated, teacher_forcing_ratio=1.0)
        
        return {
            'predicted_tokens': all_tokens,
            'sequence_probs': outputs['sequence_probs'],
            'ptm_local_presence': outputs['ptm_local_presence'],
            'ptm_local_offset': outputs['ptm_local_offset'],
            'ptm_global_presence': outputs['ptm_global_presence'],
            'ptm_global_offset': outputs['ptm_global_offset']
        }