import os
import re
import random
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


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
                    if "Mods" in spectrum:
                        spectrum["mods"] = spectrum["Mods"]
                    if "Charge" in spectrum:
                        spectrum["charge"] = spectrum["Charge"]
                    if "Parent" in spectrum:
                        spectrum["parent"] = spectrum["Parent"]
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
        if "Mods" in spectrum:
            spectrum["mods"] = spectrum["Mods"]
        if "Charge" in spectrum:
            spectrum["charge"] = spectrum["Charge"]
        if "Parent" in spectrum:
            spectrum["parent"] = spectrum["Parent"]
        yield spectrum

def preprocess_intensities(spec):
    """Apply TIC normalization and square-root transform to intensities."""
    total = spec["intensity"].sum()
    if total > 0:
        spec["intensity"] = spec["intensity"] / total
        spec["intensity"] = np.sqrt(spec["intensity"])

def apply_modifications(sequence, mods_str, ptm_dict):
    """Apply modifications to a peptide sequence and calculate global offset.
    
    Returns:
        modified_seq: The sequence with modification brackets
        global_offset: Sum of all modification masses
        ptm_details: List of (position, mass) tuples
    """
    if mods_str == "0":
        return sequence, 0.0, []
    
    # Extract the number of modifications and the modification details
    num_mods = mods_str.split("(")[0]
    if num_mods == "0":
        return sequence, 0.0, []
    
    # Parse the modifications
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, mods_str)
    
    # Transform sequence to list for easier modification
    seq_list = list(sequence)
    
    # Track detailed PTM information
    ptm_info = []
    global_offset = 0.0  # Will be sum of all PTM masses
    
    # Sort modifications by position in descending order to avoid index shifts
    all_ptms = [match.split(',') for match in matches]
    all_ptms.sort(key=lambda x: int(x[0]), reverse=True)
    
    for ptm in all_ptms:
        pos = int(ptm[0])
        aa_type = ptm[1] if len(ptm) > 1 else ""
        mod_type = ptm[2]
        
        # Handle position-specific modifications
        if mod_type in ptm_dict:
            mass_shift = ptm_dict[mod_type]
            # Add to global offset (sum of all modifications)
            global_offset += mass_shift
            
            # Insert the mass shift after the amino acid in bracket notation
            if 0 <= pos < len(seq_list):
                seq_list[pos] = seq_list[pos] + f"[+{mass_shift}]"
                ptm_info.append((pos, mass_shift))
    
    return "".join(seq_list), global_offset, ptm_info

def split_msp_data(msp_filename: str, train_ratio: float = 0.8, 
                  random_seed: Optional[int] = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Read all spectra from MSP file and split into training and test sets.
    
    Args:
        msp_filename: Path to the MSP file
        train_ratio: Ratio of data to use for training (default: 0.8)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_spectra, test_spectra)
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Read all spectra into memory
    all_spectra = list(parse_msp(msp_filename))
    
    # Shuffle the spectra
    random.shuffle(all_spectra)
    
    # Calculate split point
    split_idx = int(len(all_spectra) * train_ratio)
    
    # Split into train and test sets
    train_spectra = all_spectra[:split_idx]
    test_spectra = all_spectra[split_idx:]
    
    print(f"Split MSP data into {len(train_spectra)} training and {len(test_spectra)} test spectra")
    
    return train_spectra, test_spectra

class PTMSpectrumDataloader(Dataset):
    def __init__(self, msp_filename=None, spectra_list=None, max_peaks=200, max_seq_len=50):
        """
        Initialize dataset from either an MSP file or a list of preprocessed spectra.
        """
        self.base_sequences = []
        self.modified_sequences = []
        self.precursor_data = []
        self.peak_data = []
        self.ptm_info = []
        self.global_offsets = []
        self.max_peaks = max_peaks
        self.max_seq_len = max_seq_len
        
        # Define standard PTM masses
        self.ptm_dict = {
            "CAM": 57.02146,
            "Oxidation": 15.994915,
            "Acetyl": 42.010565,
            "Phospho": 79.966331,
            "Methyl": 14.01565,
            "Deamidation": 0.984016,
            "GlyGly": 114.042927,
        }
        
        # Create dictionary for common PTM masses with special tokens
        self.common_ptm_masses = {
            57.02146: "<CAM>",
            15.994915: "<Ox>",
            42.010565: "<Ac>",
            79.966331: "<Phos>",
            14.01565: "<Me>",
            0.984016: "<Deam>",
            114.042927: "<GG>",
        }
        
        # Define tokens for sequence processing
        self.special_tokens = {
            'PAD': '_',
            'SOS': '<sos>',
            'EOS': '<eos>',
        }
        
        # Create AA vocabulary including special tokens for modifications
        self.aa_vocab = list("_ACDEFGHIKLMNPQRSTVWY")
        self.aa_vocab.extend(['<sos>', '<eos>', '[', ']', '+', '.'])
        # Add PTM mass tokens
        self.aa_vocab.extend(self.common_ptm_masses.values())
        # Still include digits for handling any non-standard masses
        self.aa_vocab.extend([str(i) for i in range(10)])
        
        self.aa_to_id = {aa: i for i, aa in enumerate(self.aa_vocab)}
        self.id_to_aa = {v: k for k, v in self.aa_to_id.items()}
        
        # Process either from file or list
        spectra_source = []
        if msp_filename:
            spectra_source = parse_msp(msp_filename)
        elif spectra_list:
            spectra_source = spectra_list
        
        # Process spectra
        for spec in spectra_source:
            preprocess_intensities(spec)
            if "sequence" not in spec:
                continue
            
            # Get basic spectral data
            pepmass = float(spec.get("Parent", "0.0"))
            charge_value = int(spec.get("Charge", "2"))
            sequence = spec["sequence"]
            
            # Process modifications
            mods = spec.get("mods", "0")
            modified_seq, global_offset, ptm_details = apply_modifications(sequence, mods, self.ptm_dict)
            
            # Store data
            self.base_sequences.append(sequence)
            self.modified_sequences.append(modified_seq)
            self.precursor_data.append((pepmass, charge_value))
            self.ptm_info.append(ptm_details)
            self.global_offsets.append(global_offset)
            
            # Process peaks - sort by intensity and take top max_peaks
            peak_tuples = list(zip(spec["mz"], spec["intensity"]))
            peak_tuples.sort(key=lambda x: x[1], reverse=True)
            peak_tuples = peak_tuples[:self.max_peaks]
            
            # If fewer than max_peaks, pad with zeros
            if len(peak_tuples) < self.max_peaks:
                peak_tuples.extend([(0.0, 0.0)] * (self.max_peaks - len(peak_tuples)))
            
            self.peak_data.append(peak_tuples)
    
    def tokenize_modified_sequence(self, sequence):
        """
        Tokenize a sequence with modification notation, using special tokens for common PTM masses.
        Example: "PEP[+79.97]TIDE" -> ['P','E','P','[','+','<Phos>',']','T','I','D','E']
        """
        tokens = []
        i = 0
        while i < len(sequence):
            if sequence[i] == '[' and i+1 < len(sequence) and sequence[i+1] == '+':
                # Handle modification block
                end_idx = sequence.find(']', i)
                if end_idx != -1:
                    # Extract the mass value
                    mass_str = sequence[i+2:end_idx]
                    try:
                        mass_val = float(mass_str)
                        # Check if it's a common PTM mass (with slight tolerance for float comparison)
                        found = False
                        for common_mass, token in self.common_ptm_masses.items():
                            if abs(mass_val - common_mass) < 0.0001:  # Small tolerance for float comparison
                                tokens.append('[')
                                tokens.append('+')
                                tokens.append(token)
                                tokens.append(']')
                                found = True
                                break
                                
                        if not found:
                            # If not a common mass, tokenize normally
                            for char in sequence[i:end_idx+1]:
                                tokens.append(char)
                    except ValueError:
                        # If not a valid float, tokenize normally
                        for char in sequence[i:end_idx+1]:
                            tokens.append(char)
                    i = end_idx + 1
                else:
                    # If no closing bracket, just add the character
                    tokens.append(sequence[i])
                    i += 1
            else:
                tokens.append(sequence[i])
                i += 1
        return tokens
    
    def encode_tokens(self, tokens):
        """Convert tokens to IDs using vocabulary."""
        return [self.aa_to_id.get(token, self.aa_to_id['_']) for token in tokens]
    
    def prepare_sequence_tensors(self, tokens, max_len):
        """Prepare input and target tensors for the model."""
        # Truncate if too long
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        
        # Encode to IDs
        encoded = self.encode_tokens(tokens)
        
        # Create input sequence (with SOS token)
        input_tokens = [self.aa_to_id[self.special_tokens['SOS']]] + encoded[:-1] if len(encoded) == max_len else [self.aa_to_id[self.special_tokens['SOS']]] + encoded
        input_tensor = torch.tensor(input_tokens, dtype=torch.long)
        
        # Create target sequence (may include EOS token)
        target_tokens = encoded + [self.aa_to_id[self.special_tokens['EOS']]] if len(encoded) < max_len else encoded
        target_tensor = torch.tensor(target_tokens, dtype=torch.long)
        
        # Pad if needed
        if len(input_tensor) < max_len:
            padding = torch.ones(max_len - len(input_tensor), dtype=torch.long) * self.aa_to_id[self.special_tokens['PAD']]
            input_tensor = torch.cat([input_tensor, padding])
        
        if len(target_tensor) < max_len:
            padding = torch.ones(max_len - len(target_tensor), dtype=torch.long) * self.aa_to_id[self.special_tokens['PAD']]
            target_tensor = torch.cat([target_tensor, padding])
        
        return input_tensor, target_tensor
    
    def prepare_ptm_targets(self, base_seq, ptm_details, max_len):
        """Prepare PTM classification and regression targets."""
        seq_len = min(len(base_seq), max_len)
        
        # Initialize tensors for PTM targets
        ptm_presence = torch.zeros(max_len, dtype=torch.float32)
        ptm_offset = torch.zeros(max_len, dtype=torch.float32)
        
        # Fill in the tensors based on PTM details (now without type information)
        for pos, mass in ptm_details:
            if pos < max_len:
                ptm_presence[pos] = 1.0
                ptm_offset[pos] = mass
        
        return ptm_presence, ptm_offset
    
    def __len__(self):
        """The number of spectra"""
        return len(self.modified_sequences)
    
    def __getitem__(self, idx):
        """Get a single spectrum with all its associated data."""
        # Get the modified sequence and tokenize it
        modified_seq = self.modified_sequences[idx]
        tokens = self.tokenize_modified_sequence(modified_seq)
        
        # Prepare sequence tensors
        input_tensor, target_tensor = self.prepare_sequence_tensors(tokens, self.max_seq_len)
        
        # Local PTM presence/offset: shape [seq_len]
        ptm_presence, ptm_offset = self.prepare_ptm_targets(
            self.base_sequences[idx], 
            self.ptm_info[idx], 
            self.max_seq_len
        )
        
        # Global PTM: shape [1] each
        global_offset = self.global_offsets[idx]  # float
        global_ptm_presence = torch.tensor(
            [1.0 if global_offset != 0.0 else 0.0],
            dtype=torch.float32
        )
        global_ptm_offset = torch.tensor([global_offset], dtype=torch.float32)
        
        # Prepare MS2 spectrum peaks
        peak_data = self.peak_data[idx]
        mz_values = torch.tensor([p[0] for p in peak_data], dtype=torch.float32)
        intensity_values = torch.tensor([p[1] for p in peak_data], dtype=torch.float32)
        
        # Prepare precursor data
        precursor_mass, charge = self.precursor_data[idx]
        precursor_tensor = torch.tensor([precursor_mass, float(charge)], dtype=torch.float32)
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attn_mask = torch.ones(self.max_seq_len, dtype=torch.bool)
        token_len = min(len(tokens), self.max_seq_len)
        if token_len < self.max_seq_len:
            attn_mask[token_len:] = False
        
        return {
            'precursor': precursor_tensor,
            'peaks': (mz_values, intensity_values),
            'input_sequence': input_tensor,
            'base_sequence': self.base_sequences[idx],
            'modified_sequence': modified_seq,
            'attention_mask': attn_mask,
            'sequence_length': token_len,
        }, {
            'target_sequence': target_tensor,
            'ptm_presence': ptm_presence,      # shape [seq_len]
            'ptm_offset': ptm_offset,          # shape [seq_len]
            'global_ptm_presence': global_ptm_presence,  # shape [1]
            'global_ptm_offset': global_ptm_offset       # shape [1]
        }
    
def create_train_test_dataloaders(msp_filename, train_ratio=0.8, max_peaks=200, max_seq_len=50, 
                                 batch_size=32, random_seed=42):
    """
    Create training and test dataloaders from an MSP file.
    
    Args:
        msp_filename: Path to the MSP file
        train_ratio: Ratio of data to use for training (default: 0.8)
        max_peaks: Maximum number of peaks to use per spectrum
        max_seq_len: Maximum sequence length
        batch_size: Batch size for dataloaders
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Split the data
    train_spectra, test_spectra = split_msp_data(msp_filename, train_ratio, random_seed)
    
    # Create datasets
    train_dataset = PTMSpectrumDataloader(spectra_list=train_spectra, max_peaks=max_peaks, max_seq_len=max_seq_len)
    test_dataset = PTMSpectrumDataloader(spectra_list=test_spectra, max_peaks=max_peaks, max_seq_len=max_seq_len)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, test_loader