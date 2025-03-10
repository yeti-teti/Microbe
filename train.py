import os
import time
import argparse
import math  # Added missing import
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from rich.console import Console
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.table import Table

from data_loader import create_train_test_dataloaders
from model import Spec2PepWithPTM

console = Console()

def parse_args():
    parser = argparse.ArgumentParser(description="Train Spec2Pep model with PTM prediction")
    
    # Data parameters
    parser.add_argument("--msp_file", type=str, required=True, help="Path to MSP file")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--max_peaks", type=int, default=200, help="Maximum number of peaks to use")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Maximum sequence length")
    
    # Model parameters
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Initial learning rate")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps for learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument("--seq_loss_weight", type=float, default=1.0, help="Weight for sequence loss")
    parser.add_argument("--ptm_local_loss_weight", type=float, default=0.5, help="Weight for local PTM loss")
    parser.add_argument("--ptm_global_loss_weight", type=float, default=0.5, help="Weight for global PTM loss")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Maximum gradient norm")
    
    # Scheduled sampling parameters
    parser.add_argument("--teacher_forcing_start", type=float, default=1.0, 
                        help="Initial teacher forcing ratio")
    parser.add_argument("--teacher_forcing_end", type=float, default=0.5, 
                        help="Final teacher forcing ratio")
    
    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.8, 
                        help="Temperature for sequence generation")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--log_interval", type=int, default=10, help="Logging interval (batches)")
    parser.add_argument("--save_interval", type=int, default=1, help="Saving interval (epochs)")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    
    return parser.parse_args()

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_loss(outputs, targets, loss_weights, pad_idx=0, label_smoothing=0.1):
    """
    outputs dict: {
        'sequence_probs': [batch, seq_len, vocab_size],
        'ptm_local_presence': [batch, seq_len],  (logits)
        'ptm_local_offset': [batch, seq_len],
        'ptm_global_presence': [batch],          (logits)
        'ptm_global_offset': [batch]
    }
    targets dict: {
        'target_sequence': [batch, seq_len],
        'ptm_presence': [batch, seq_len],   (0 or 1)
        'ptm_offset': [batch, seq_len],
        'global_ptm_presence': [batch,1] or [batch],
        'global_ptm_offset': [batch,1] or [batch]
    }
    """
    # 1) Sequence prediction loss
    seq_loss = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=label_smoothing
    )(
        outputs['sequence_probs'].view(-1, outputs['sequence_probs'].size(-1)),
        targets['target_sequence'].view(-1)
    )
    
    # 2) Local PTM presence -> raw logits + BCEWithLogits
    ptm_local_presence_logits = outputs['ptm_local_presence']        # [batch, seq_len]
    ptm_local_presence_target = targets['ptm_presence'].float()      # [batch, seq_len]
    
    ptm_local_presence_loss = nn.BCEWithLogitsLoss()(
        ptm_local_presence_logits.view(-1),   # flatten
        ptm_local_presence_target.view(-1)
    )
    
    # 3) Local PTM offset -> MSE only where presence=1
    ptm_mask = ptm_local_presence_target  # shape [batch, seq_len], 0 or 1
    ptm_local_offset = outputs['ptm_local_offset']  # [batch, seq_len]
    ptm_offset_target = targets['ptm_offset'].float()  # [batch, seq_len]
    
    ptm_local_offset_loss = nn.MSELoss()(
        ptm_local_offset * ptm_mask,   # zero out non-PTM positions
        ptm_offset_target * ptm_mask
    )
    
    # 4) Global PTM presence -> raw logits + BCEWithLogits
    global_ptm_presence_logits = outputs['ptm_global_presence']  # [batch] or [batch,1]
    global_ptm_presence_target = targets['global_ptm_presence'].float()
    if global_ptm_presence_target.dim() == 2 and global_ptm_presence_target.size(1) == 1:
        # Squeeze to [batch]
        global_ptm_presence_target = global_ptm_presence_target.squeeze(-1)
    if global_ptm_presence_logits.dim() == 2 and global_ptm_presence_logits.size(1) == 1:
        global_ptm_presence_logits = global_ptm_presence_logits.squeeze(-1)
    
    ptm_global_presence_loss = nn.BCEWithLogitsLoss()(
        global_ptm_presence_logits,   # [batch]
        global_ptm_presence_target    # [batch]
    )
    
    # 5) Global offset -> MSE only if presence=1
    global_ptm_offset_output = outputs['ptm_global_offset']  # [batch] or [batch,1]
    global_ptm_offset_target = targets['global_ptm_offset'].float()  # [batch] or [batch,1]
    
    # Squeeze if needed
    if global_ptm_offset_output.dim() == 2 and global_ptm_offset_output.size(1) == 1:
        global_ptm_offset_output = global_ptm_offset_output.squeeze(-1)
    if global_ptm_offset_target.dim() == 2 and global_ptm_offset_target.size(1) == 1:
        global_ptm_offset_target = global_ptm_offset_target.squeeze(-1)
    
    # Multiply by presence=1 to zero out negative cases
    masked_global_offset_output = global_ptm_offset_output * global_ptm_presence_target
    masked_global_offset_target = global_ptm_offset_target * global_ptm_presence_target
    
    ptm_global_offset_loss = nn.MSELoss()(
        masked_global_offset_output,
        masked_global_offset_target
    )
    
    # Combine losses
    total_loss = (
        loss_weights['seq'] * seq_loss
        + loss_weights['ptm_local'] * (ptm_local_presence_loss + ptm_local_offset_loss)
        + loss_weights['ptm_global'] * (ptm_global_presence_loss + ptm_global_offset_loss)
    )
    
    loss_info = {
        'total': total_loss.item(),
        'seq': seq_loss.item(),
        'ptm_local_presence': ptm_local_presence_loss.item(),
        'ptm_local_offset': ptm_local_offset_loss.item(),
        'ptm_global_presence': ptm_global_presence_loss.item(),
        'ptm_global_offset': ptm_global_offset_loss.item()
    }
    
    return total_loss, loss_info


def get_lr_scheduler(optimizer, warmup_steps=2000, total_steps=100000):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    
    Args:
        optimizer: The optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(step):
        # Linear warmup
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        
        # Cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)

def train_one_epoch(
    model, 
    dataloader, 
    optimizer, 
    scheduler, 
    loss_weights, 
    device, 
    pad_idx=0, 
    log_interval=10, 
    max_grad_norm=1.0,
    label_smoothing=0.1,
    teacher_forcing_ratio=1.0,
    temperature=1.0
):
    """
    Train the model for one epoch.
    
    Args:
        model: The model to train
        dataloader: DataLoader with training data
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        loss_weights: Dictionary of loss weights
        device: Device to use
        pad_idx: Index of padding token
        log_interval: How often to log progress
        max_grad_norm: Maximum gradient norm for clipping
        label_smoothing: Label smoothing factor
        teacher_forcing_ratio: Probability of using teacher forcing
        temperature: Temperature for sequence generation
        
    Returns:
        avg_loss: Average loss over the epoch
        loss_info: Dictionary of average loss components
    """
    model.train()
    total_loss = 0
    loss_info_sum = {
        'total': 0,
        'seq': 0,
        'ptm_local_presence': 0, 
        'ptm_local_offset': 0,
        'ptm_global_presence': 0,
        'ptm_global_offset': 0
    }
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn()
    ) as progress:
        train_task = progress.add_task("Training", total=len(dataloader))
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            # Move features to device
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                      (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                      for k, v in features.items()}
            
            # Move targets to device
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(
                features, 
                teacher_forcing_ratio=teacher_forcing_ratio,
                temperature=temperature
            )
            
            # Calculate loss
            loss, batch_loss_info = calculate_loss(
                outputs, 
                targets, 
                loss_weights, 
                pad_idx, 
                label_smoothing
            )
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses
            for k, v in batch_loss_info.items():
                loss_info_sum[k] += v
            
            # Update progress
            progress.update(train_task, advance=1)
            
            # Log progress (only basic info during batches)
            if batch_idx % log_interval == 0:
                progress.console.log(f"Batch {batch_idx}/{len(dataloader)}, Loss: {batch_loss_info['total']:.4f}")
    
    # Calculate averages
    batch_count = len(dataloader)
    avg_loss_info = {k: v / batch_count for k, v in loss_info_sum.items()}
    
    return avg_loss_info['total'], avg_loss_info


def evaluate(
    model, 
    dataloader, 
    loss_weights, 
    device, 
    pad_idx=0, 
    label_smoothing=0.1,
    temperature=1.0
):
    """
    Evaluate the model on the given dataloader.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader to use for evaluation
        loss_weights: Dictionary of loss weights
        device: Device to use
        pad_idx: Index of padding token
        label_smoothing: Label smoothing factor
        temperature: Temperature for sequence generation
        
    Returns:
        avg_loss: Average loss over the dataset
        loss_info: Dictionary of average loss components
    """
    model.eval()
    total_loss = 0
    loss_info_sum = {
        'total': 0,
        'seq': 0,
        'ptm_local_presence': 0, 
        'ptm_local_offset': 0,
        'ptm_global_presence': 0,
        'ptm_global_offset': 0
    }
    
    with torch.no_grad():
        for features, targets in dataloader:
            # Move features to device
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                      (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                      for k, v in features.items()}
            
            # Move targets to device
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass
            outputs = model(features, teacher_forcing_ratio=1.0, temperature=temperature)
            
            # Calculate loss
            _, batch_loss_info = calculate_loss(
                outputs, 
                targets, 
                loss_weights, 
                pad_idx, 
                label_smoothing
            )
            
            # Accumulate losses
            for k, v in batch_loss_info.items():
                loss_info_sum[k] += v
    
    # Calculate averages
    batch_count = len(dataloader)
    avg_loss_info = {k: v / batch_count for k, v in loss_info_sum.items()}
    
    return avg_loss_info['total'], avg_loss_info

def decode_sequence(id_to_aa, sequence_ids):
    """
    Decode a sequence from token IDs to amino acid string.
    
    Args:
        id_to_aa: Dictionary mapping token IDs to amino acids
        sequence_ids: Tensor of token IDs
        
    Returns:
        Decoded amino acid sequence
    """
    # Convert tensor to list of integers
    if isinstance(sequence_ids, torch.Tensor):
        sequence_ids = sequence_ids.cpu().tolist()
    
    # Decode to amino acids
    amino_acids = [id_to_aa.get(idx, '_') for idx in sequence_ids]
    
    # Remove special tokens
    valid_aas = []
    for aa in amino_acids:
        if aa in ['<sos>', '<eos>', '_']:
            continue
        valid_aas.append(aa)
    
    return ''.join(valid_aas)

def analyze_predictions(console, model, features, targets, id_to_aa, max_samples=5):
    """
    Analyze model predictions to check diversity.
    
    Args:
        console: Rich console for output
        model: The model
        features: Batch features
        targets: Batch targets
        id_to_aa: ID to amino acid mapping
        max_samples: Maximum number of samples to display (default: 5)
    """
    model.eval()
    
    with torch.no_grad():
        # Get device
        device = next(model.parameters()).device
        
        # Move data to device
        features_device = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                          (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                          for k, v in features.items()}
        
        # Forward pass with different temperatures
        outputs_cold = model(features_device, teacher_forcing_ratio=0.0, temperature=0.5)
        outputs_normal = model(features_device, teacher_forcing_ratio=0.0, temperature=1.0)
        
        # Get predictions
        pred_cold = outputs_cold['sequence_probs'].argmax(dim=-1)  # [batch, seq_len]
        pred_normal = outputs_normal['sequence_probs'].argmax(dim=-1)  # [batch, seq_len]
        
        # Count unique amino acids for the first few examples
        n_examples = min(max_samples, len(pred_cold))
        
        aa_counts = {}
        console.print("\n[bold]Sample Predictions:[/bold]")
        for i in range(n_examples):
            seq_cold = decode_sequence(id_to_aa, pred_cold[i])
            seq_normal = decode_sequence(id_to_aa, pred_normal[i])
            true_seq = decode_sequence(id_to_aa, targets['target_sequence'][i])
            
            for aa in seq_cold:
                if aa not in aa_counts:
                    aa_counts[aa] = 0
                aa_counts[aa] += 1
            
            console.print(f"Example {i+1}:")
            console.print(f"  True:    {true_seq}")
            console.print(f"  Cold:    {seq_cold}")
            console.print(f"  Normal:  {seq_normal}")
            
            # Add PTM information
            true_ptm_pos = targets['ptm_presence'][i].nonzero().squeeze(-1).cpu().tolist()
            if not isinstance(true_ptm_pos, list):
                true_ptm_pos = [true_ptm_pos]
            
            pred_ptm_pos = (outputs_normal['ptm_local_presence'][i] > 0.5).nonzero().squeeze(-1).cpu().tolist()
            if not isinstance(pred_ptm_pos, list):
                pred_ptm_pos = [pred_ptm_pos]
            
            console.print(f"  True PTM positions:  {true_ptm_pos}")
            console.print(f"  Pred PTM positions:  {pred_ptm_pos}")
            console.print("")
    
    console.print(f"Amino acid distribution in predictions: {sorted(aa_counts.items())}")

def predict_and_visualize(model, dataloader, id_to_aa, device, num_samples=5):
    """
    Make predictions and visualize results for a few samples.
    
    Args:
        model: The trained model
        dataloader: DataLoader to use
        id_to_aa: Dictionary mapping token IDs to amino acids
        device: Device to use for tensors
        num_samples: Number of samples to visualize
        
    Returns:
        Table with visualization results
    """
    model.eval()
    samples = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            if len(samples) >= num_samples:
                break
                
            # Move data to device
            features = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                       (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                       for k, v in features.items()}
            targets = {k: v.to(device) for k, v in targets.items()}
            
            # Forward pass with lower temperature
            outputs = model(features, teacher_forcing_ratio=0.0, temperature=0.7)
            
            # Get predictions
            pred_sequence = outputs['sequence_probs'].argmax(dim=-1)  # [batch, seq_len]
            pred_ptm_presence = (outputs['ptm_local_presence'] > 0.5).float()  # [batch, seq_len]
            
            # Process batch
            for i in range(min(len(pred_sequence), num_samples - len(samples))):
                # Get true sequence
                true_seq = decode_sequence(id_to_aa, targets['target_sequence'][i])
                
                # Get predicted sequence
                pred_seq = decode_sequence(id_to_aa, pred_sequence[i])
                
                # Get predicted PTMs
                pred_ptm_pos = pred_ptm_presence[i].nonzero().squeeze(-1).cpu().tolist()
                if not isinstance(pred_ptm_pos, list):
                    pred_ptm_pos = [pred_ptm_pos]
                pred_ptm_offsets = [f"{outputs['ptm_local_offset'][i, pos].item():.2f}" for pos in pred_ptm_pos]
                
                # Get true PTMs
                true_ptm_pos = targets['ptm_presence'][i].nonzero().squeeze(-1).cpu().tolist()
                if not isinstance(true_ptm_pos, list):
                    true_ptm_pos = [true_ptm_pos]
                true_ptm_offsets = [f"{targets['ptm_offset'][i, pos].item():.2f}" for pos in true_ptm_pos]
                
                # Get global PTM info
                pred_global_presence = outputs['ptm_global_presence'][i].item() > 0.5
                pred_global_offset = outputs['ptm_global_offset'][i].item()
                true_global_presence = targets['global_ptm_presence'][i].item() > 0.5
                true_global_offset = targets['global_ptm_offset'][i].item()
                
                # Add to samples
                samples.append({
                    'true_seq': true_seq,
                    'pred_seq': pred_seq,
                    'true_ptm_pos': true_ptm_pos,
                    'true_ptm_offsets': true_ptm_offsets,
                    'pred_ptm_pos': pred_ptm_pos,
                    'pred_ptm_offsets': pred_ptm_offsets,
                    'true_global_ptm': (true_global_presence, true_global_offset),
                    'pred_global_ptm': (pred_global_presence, pred_global_offset)
                })
    
    # Create visualization table
    table = Table(title="Prediction Samples")
    table.add_column("Sample", justify="center")
    table.add_column("True Sequence", justify="left")
    table.add_column("Predicted Sequence", justify="left")
    table.add_column("True PTMs", justify="left")
    table.add_column("Predicted PTMs", justify="left")
    table.add_column("True Global PTM", justify="center")
    table.add_column("Predicted Global PTM", justify="center")
    
    for i, sample in enumerate(samples):
        true_ptms = ", ".join([f"Pos {pos}: {offset}" for pos, offset in zip(sample['true_ptm_pos'], sample['true_ptm_offsets'])])
        pred_ptms = ", ".join([f"Pos {pos}: {offset}" for pos, offset in zip(sample['pred_ptm_pos'], sample['pred_ptm_offsets'])])
        
        true_global = f"{'Yes' if sample['true_global_ptm'][0] else 'No'} ({sample['true_global_ptm'][1]:.2f})"
        pred_global = f"{'Yes' if sample['pred_global_ptm'][0] else 'No'} ({sample['pred_global_ptm'][1]:.2f})"
        
        table.add_row(
            str(i+1),
            sample['true_seq'],
            sample['pred_seq'],
            true_ptms if true_ptms else "None",
            pred_ptms if pred_ptms else "None",
            true_global,
            pred_global
        )
    
    return table

def save_checkpoint(model, optimizer, scheduler, epoch, loss_info, filename):
    """Save checkpoint to file."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss_info': loss_info
    }
    torch.save(checkpoint, filename)
    console.log(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, scheduler, filename):
    """Load checkpoint from file."""
    console.log(f"Loading checkpoint from {filename}")
    
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    start_epoch = checkpoint['epoch'] + 1
    console.log(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    return start_epoch

def train_model(args):
    """Main training function."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        console.log("[yellow]CUDA not available, using CPU instead[/yellow]")
        device = torch.device("cpu")
    elif args.device == "mps" and not torch.backends.mps.is_available():
        console.log("[yellow]MPS not available, using CPU instead[/yellow]")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    console.log(f"Using device: {device}")
    
    # Set random seed
    set_seed(args.seed)
    
    # Create dataloaders
    console.log("Loading data...")
    train_loader, val_loader = create_train_test_dataloaders(
        args.msp_file,
        train_ratio=0.8,
        max_peaks=args.max_peaks,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        random_seed=args.seed
    )
    
    # Get vocabulary size from the dataloader
    vocab_size = len(train_loader.dataset.aa_to_id)
    id_to_aa = train_loader.dataset.id_to_aa
    pad_idx = train_loader.dataset.aa_to_id['_']
    
    console.log(f"Vocabulary size: {vocab_size}")
    
    # Create model
    console.log("Creating model...")
    model = Spec2PepWithPTM(
        vocab_size=vocab_size,
        max_peaks=args.max_peaks,
        max_seq_len=args.max_seq_len,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Calculate total steps for scheduler
    total_steps = len(train_loader) * args.epochs
    
    # Setup scheduler with warmup
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    
    # Setup tensorboard
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Initialize loss weights
    loss_weights = {
        'seq': args.seq_loss_weight,
        'ptm_local': args.ptm_local_loss_weight,
        'ptm_global': args.ptm_global_loss_weight
    }
    
    # Training loop
    console.log("Starting training...")
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Check if resume from checkpoint
    checkpoint_path = os.path.join(args.output_dir, 'checkpoint.pt')
    if args.resume and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    for epoch in range(start_epoch, args.epochs):
        console.log(f"[bold]Epoch {epoch+1}/{args.epochs}[/bold]")
        
        # Calculate teacher forcing ratio for this epoch using linear annealing
        teacher_forcing_ratio = max(
            args.teacher_forcing_end,
            args.teacher_forcing_start - (args.teacher_forcing_start - args.teacher_forcing_end) * 
            (epoch / args.epochs)
        )
        console.log(f"Teacher forcing ratio: {teacher_forcing_ratio:.2f}")
        
        # Train for one epoch
        train_loss, train_loss_info = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_weights, device,
            pad_idx=pad_idx, 
            log_interval=args.log_interval,
            max_grad_norm=args.max_grad_norm,
            label_smoothing=args.label_smoothing,
            teacher_forcing_ratio=teacher_forcing_ratio,
            temperature=args.temperature
        )
        
        # Evaluate on validation set
        val_loss, val_loss_info = evaluate(
            model, val_loader, loss_weights, device, 
            pad_idx=pad_idx,
            label_smoothing=args.label_smoothing,
            temperature=args.temperature
        )
        
        # Log metrics
        console.log(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        console.log("Loss Components:")
        for k, v in train_loss_info.items():
            if k == 'total':
                continue
            console.log(f"  Train {k}: {v:.4f}, Val {k}: {val_loss_info[k]:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Teacher_Forcing_Ratio', teacher_forcing_ratio, epoch)
        
        for k, v in train_loss_info.items():
            if k == 'total':
                continue
            writer.add_scalar(f'LossComponents/train_{k}', v, epoch)
        
        for k, v in val_loss_info.items():
            if k == 'total':
                continue
            writer.add_scalar(f'LossComponents/val_{k}', v, epoch)
        
        # Analyze predictions only after each epoch
        # Get a small batch for analysis
        eval_features, eval_targets = next(iter(val_loader))
        analyze_predictions(console, model, eval_features, eval_targets, id_to_aa, max_samples=5)
        
        # Save checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_loss_info, 'val': val_loss_info},
                os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_loss_info, 'val': val_loss_info},
                os.path.join(args.output_dir, 'best_model.pt')
            )
            console.log("[green]New best model saved![/green]")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                console.log(f"[yellow]Early stopping after {patience_counter} epochs without improvement[/yellow]")
                break
        
        # Save latest checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch,
            {'train': train_loss_info, 'val': val_loss_info},
            checkpoint_path
        )
        
        # Show some predictions
        if (epoch + 1) % args.save_interval == 0:
            with torch.no_grad():
                table = predict_and_visualize(model, val_loader, id_to_aa, device, num_samples=5)
                console.print(table)
    
    console.log("[green]Training complete![/green]")
    
    # Final evaluation
    console.log("Final evaluation...")
    test_loss, test_loss_info = evaluate(
        model, val_loader, loss_weights, device, 
        pad_idx=pad_idx,
        label_smoothing=args.label_smoothing
    )
    console.log(f"Test Loss: {test_loss:.4f}")
    
    # Visualize final predictions
    with torch.no_grad():
        table = predict_and_visualize(model, val_loader, id_to_aa, device, num_samples=5)
        console.print(table)
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1,
        {'val': test_loss_info},
        os.path.join(args.output_dir, 'final_model.pt')
    )
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)