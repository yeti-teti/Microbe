import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.progress import track
import time

# Global token definitions
PAD_TOKEN_ID = 0    # Padding token ID
SOS_TOKEN_ID = 1    # Start-of-sequence token ID
EOS_TOKEN_ID = 2    # End-of-sequence token ID

# Define the amino acid vocabulary size.
# Vocabulary: PAD, SOS, EOS, then 20 standard amino acids.
AA_VOCAB_SIZE = 23  

# Define offset bins: offsets from -10 to +100 (inclusive)
OFFSET_MIN = -10
OFFSET_MAX = 100
NUM_OFFSET_BINS = OFFSET_MAX - OFFSET_MIN + 1  # e.g., 111 bins

# -------------------------------------------------------------------
# Data Preprocessing: MSP Parsing and Spectrum Structuring
# -------------------------------------------------------------------
def parse_msp(filename):
    """Parse an MSP spectral library file and yield spectra as dictionaries."""
    spectrum = {}
    peaks_mz, peaks_int, peaks_ann = [], [], []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Name: "):
                if spectrum:
                    spectrum["mz"] = np.array(peaks_mz, dtype=np.float32)
                    spectrum["intensity"] = np.array(peaks_int, dtype=np.float32)
                    if "Fullname" in spectrum:
                        spectrum["sequence"] = spectrum["Fullname"].split(".")[1]
                    yield spectrum
                spectrum = {"Name": line[6:].strip()}
                peaks_mz, peaks_int, peaks_ann = [], [], []
            elif line.startswith("Comment: "):
                for item in line[9:].split():
                    if "=" in item:
                        key, val = item.split("=", 1)
                        spectrum[key] = val
            elif line.startswith("Num peaks:"):
                spectrum["NumPeaks"] = int(line.split(":")[1].strip())
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz_val = float(parts[0])
                        intensity_val = float(parts[1])
                    except ValueError:
                        continue
                    peaks_mz.append(mz_val)
                    peaks_int.append(intensity_val)
                    if len(parts) > 2:
                        ann = parts[2].strip('"')
                        peaks_ann.append(ann)
    if spectrum:
        spectrum["mz"] = np.array(peaks_mz, dtype=np.float32)
        spectrum["intensity"] = np.array(peaks_int, dtype=np.float32)
        if "Fullname" in spectrum:
            spectrum["sequence"] = spectrum["Fullname"].split(".")[1]
        yield spectrum

# -------------------------------------------------------------------
# Intensity Normalization and Transformation
# -------------------------------------------------------------------
def preprocess_intensities(spec):
    """Apply TIC normalization and square-root transform to intensities."""
    total = spec["intensity"].sum()
    if total > 0:
        spec["intensity"] = spec["intensity"] / total
        spec["intensity"] = np.sqrt(spec["intensity"])

# -------------------------------------------------------------------
# MSP to MGF Converter
# -------------------------------------------------------------------
def msp_to_mgf(msp_filename, mgf_filename):
    """Convert an MSP file to an annotated MGF file for Casanovo training."""
    out = open(mgf_filename, "w")
    spectrum_index = 0
    for spec in parse_msp(msp_filename):
        preprocess_intensities(spec)
        if "sequence" not in spec:
            continue
        pepmass = spec.get("PEPMASS", "0.0")
        charge = spec.get("Charge", "2")
        if not charge.endswith("+"):
            charge = charge + "+"
        out.write("BEGIN IONS\n")
        out.write(f"TITLE=Spec_{spectrum_index}\n")
        out.write(f"PEPMASS={pepmass}\n")
        out.write(f"CHARGE={charge}\n")
        out.write(f"SEQ={spec['sequence']}\n")
        for mz, inten in zip(spec["mz"], spec["intensity"]):
            out.write(f"{mz:.4f} {inten:.4f}\n")
        out.write("END IONS\n\n")
        spectrum_index += 1
    out.close()
    print(f"Converted MSP '{msp_filename}' to MGF '{mgf_filename}'.")

# -------------------------------------------------------------------
# Casanovo Transformer Model for PTM Prediction (batch_first=True)
# -------------------------------------------------------------------
class CasanovoPTM(nn.Module):
    def __init__(self, aa_vocab_size, num_offset_bins, d_model=256, nhead=2, num_layers=2):
        super(CasanovoPTM, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.peak_emb = nn.Linear(2, d_model)
        self.pos_encoder = nn.Embedding(5000, d_model)
        self.aa_emb = nn.Embedding(aa_vocab_size, d_model)
        self.fc_aa = nn.Linear(d_model, aa_vocab_size)
        self.fc_mod = nn.Linear(d_model, num_offset_bins)
    
    def forward(self, peak_seq, peak_mask, tgt_seq, tgt_mask):
        """
        peak_seq: [batch, src_len, 2] tensor of (mz, intensity)
        tgt_seq: [batch, tgt_len] tensor of token indices
        """
        batch_size, src_len, _ = peak_seq.size()
        pos_idx = torch.arange(src_len, device=peak_seq.device).unsqueeze(0).expand(batch_size, src_len)
        enc_in = self.peak_emb(peak_seq) + self.pos_encoder(pos_idx)
        enc_out = self.encoder(enc_in, src_key_padding_mask=peak_mask)
        
        batch_size, tgt_len = tgt_seq.size()
        pos_tgt = torch.arange(tgt_len, device=tgt_seq.device).unsqueeze(0).expand(batch_size, tgt_len)
        tgt_emb = self.aa_emb(tgt_seq) + self.pos_encoder(pos_tgt)
        dec_out = self.decoder(tgt_emb, enc_out, tgt_mask=tgt_mask, memory_key_padding_mask=peak_mask)
        aa_logits = self.fc_aa(dec_out)
        mod_logits = self.fc_mod(dec_out)
        return aa_logits, mod_logits

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz, device=next(self.parameters()).device), diagonal=1).bool()
        mask = mask.float().masked_fill(mask, float('-inf'))
        return mask

# -------------------------------------------------------------------
# PyTorch Dataset for Annotated MGF Files (only PTM peptides)
# -------------------------------------------------------------------
from torch.utils.data import Dataset, DataLoader

class MGF_Dataset(Dataset):
    def __init__(self, mgf_filename, src_len=100, tgt_len=36):
        self.spectra = self.load_mgf(mgf_filename)
        # Filter spectra to include only peptides with PTMs
        self.spectra = [spec for spec in self.spectra if spec.get("Mods", "0") != "0"]
        print(f"Filtered dataset size (with PTMs): {len(self.spectra)}")
        self.src_len = src_len
        self.tgt_len = tgt_len
    
    def load_mgf(self, filename):
        spectra = []
        current_spec = {}
        peaks = []
        with open(filename, "r") as f:
            for line in f:
                line = line.strip()
                if line == "BEGIN IONS":
                    current_spec = {}
                    peaks = []
                elif line.startswith("TITLE="):
                    current_spec["title"] = line.split("=", 1)[1]
                elif line.startswith("PEPMASS="):
                    current_spec["pepmass"] = float(line.split("=")[1])
                elif line.startswith("CHARGE="):
                    ch = line.split("=", 1)[1].replace("+", "")
                    current_spec["charge"] = int(ch)
                elif line.startswith("SEQ="):
                    current_spec["sequence"] = line.split("=", 1)[1]
                elif line.startswith("Mods="):
                    current_spec["Mods"] = line.split("=", 1)[1]
                elif line == "END IONS":
                    current_spec["mz"] = []
                    current_spec["intensity"] = []
                    for peak in peaks:
                        parts = peak.split()
                        current_spec["mz"].append(float(parts[0]))
                        current_spec["intensity"].append(float(parts[1]))
                    current_spec["mz"] = np.array(current_spec["mz"], dtype=np.float32)
                    current_spec["intensity"] = np.array(current_spec["intensity"], dtype=np.float32)
                    spectra.append(current_spec)
                else:
                    if line:
                        peaks.append(line)
        return spectra
    
    def __len__(self):
        return len(self.spectra)
    
    def __getitem__(self, idx):
        spec = self.spectra[idx]
        order = np.argsort(spec["mz"])
        mz = spec["mz"][order]
        intensity = spec["intensity"][order]
        peak_tensor = torch.tensor(np.stack([mz, intensity], axis=1), dtype=torch.float32)
        if peak_tensor.size(0) < self.src_len:
            pad = torch.zeros(self.src_len - peak_tensor.size(0), 2)
            peak_tensor = torch.cat([peak_tensor, pad], dim=0)
            peak_mask = torch.zeros(self.src_len, dtype=torch.bool)
        else:
            peak_tensor = peak_tensor[:self.src_len]
            peak_mask = torch.zeros(self.src_len, dtype=torch.bool)
        # Normalize target sequence: uppercase and strip spaces.
        seq = spec["sequence"].strip().upper()
        # Allowed vocabulary: <PAD>, <SOS>, <EOS>, then 20 standard amino acids.
        vocab = ["<PAD>", "<SOS>", "<EOS>"] + list("ACDEFGHIKLMNPQRSTVWY")
        aa_to_idx = {token: idx for idx, token in enumerate(vocab)}
        tgt_tokens = [aa_to_idx["<SOS>"]]
        for ch in seq:
            tgt_tokens.append(aa_to_idx.get(ch, aa_to_idx["A"]))
        tgt_tokens.append(aa_to_idx["<EOS>"])
        if len(tgt_tokens) < self.tgt_len:
            tgt_tokens = tgt_tokens + [PAD_TOKEN_ID]*(self.tgt_len - len(tgt_tokens))
        else:
            tgt_tokens = tgt_tokens[:self.tgt_len]
        tgt_tensor = torch.tensor(tgt_tokens, dtype=torch.long)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.tgt_len)
        return peak_tensor, peak_mask, tgt_tensor, tgt_mask

def get_dataloaders(mgf_filename, batch_size=4, split_ratio=0.8):
    dataset = MGF_Dataset(mgf_filename)
    total = len(dataset)
    train_size = int(split_ratio * total)
    val_size = total - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# -------------------------------------------------------------------
# Training Loop with Mixed Precision (AMP) and Rich Progress
# -------------------------------------------------------------------
def train_model(model, train_loader, val_loader, num_epochs=10, device=torch.device("cuda")):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scaler = torch.amp.GradScaler(device_type="cuda")
    for epoch in range(num_epochs):
        total_loss = 0.0
        # Use rich's progress track to monitor batches.
        for i, (peak_seq, peak_mask, tgt_seq, tgt_mask) in enumerate(track(train_loader, description=f"Epoch {epoch+1}/{num_epochs}")):
            peak_seq = peak_seq.to(device)
            peak_mask = peak_mask.to(device)
            tgt_seq = tgt_seq.to(device)
            tgt_mask = tgt_mask.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                # Unbatch target mask: all examples use the same mask.
                mask = tgt_mask[0, :-1, :-1]
                aa_logits, mod_logits = model(peak_seq, peak_mask, tgt_seq[:, :-1], mask)
                aa_target = tgt_seq[:, 1:]
                # For unmodified peptides, set mod target to index corresponding to 0 offset.
                mod_target = torch.full_like(tgt_seq[:, 1:], 10)
                loss_aa = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(aa_logits.reshape(-1, AA_VOCAB_SIZE), aa_target.reshape(-1))
                loss_mod = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)(mod_logits.reshape(-1, NUM_OFFSET_BINS), mod_target.reshape(-1))
                loss = loss_aa + loss_mod
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_loss:.4f}")
    torch.save(model.state_dict(), "casanovo_ptm_finetuned.pt")
    print("Training complete. Model saved as 'casanovo_ptm_finetuned.pt'.")

if __name__ == "__main__":
    print("Checking device")
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    
    print("Loading the data...")
    msp_file = "./datasets/human_hcd_tryp_best.msp"
    mgf_file = "./datasets/human_hcd_tryp_best.mgf"
    if not os.path.exists(mgf_file):
        msp_to_mgf(msp_file, mgf_file)
    print("Data loaded.")
    
    print("Initializing the dataloaders..")
    train_loader, val_loader = get_dataloaders(mgf_file, batch_size=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CasanovoPTM(AA_VOCAB_SIZE, NUM_OFFSET_BINS, d_model=256, nhead=2, num_layers=2).to(device)
    
    print("Load Model")
    checkpoint_path = "models/casanovo_v4_2_0.ckpt"
    if os.path.exists(checkpoint_path):
        import numpy as np
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print("Loaded pretrained Casanovo weights from", checkpoint_path)
    else:
        print("Pretrained checkpoint not found. Exiting.")
        exit(1)
    
    print("Training Started...")
    train_model(model, train_loader, val_loader, num_epochs=1, device=device)
