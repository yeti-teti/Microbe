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
    
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

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
        
        # Initialize parameters
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, features):
        """
        Forward pass through the model.
        
        Args:
            features: Dictionary containing:
                - 'peaks': Tuple of (mz_values, intensity_values)
                - 'precursor': Precursor info tensor [batch, 2]
                - 'input_sequence': Input sequence token IDs [batch, seq_len]
                - 'attention_mask': Boolean mask for sequence (True = use, False = ignore)
        
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
        
        # Generate outputs
        sequence_probs = self.sequence_generator(decoder_output)
        ptm_local_presence, ptm_local_offset = self.ptm_local_predictor(decoder_output)
        ptm_global_presence, ptm_global_offset = self.ptm_global_predictor(
            decoder_output, 
            mask=features.get('attention_mask', None).to(device) if features.get('attention_mask', None) is not None else None
        )
        
        return {
            'sequence_probs': sequence_probs,
            'ptm_local_presence': ptm_local_presence,
            'ptm_local_offset': ptm_local_offset,
            'ptm_global_presence': ptm_global_presence,
            'ptm_global_offset': ptm_global_offset
        }