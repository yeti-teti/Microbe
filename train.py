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

def calculate_loss(outputs, targets, loss_weights, pad_idx=0, label_smoothing=0.1):
    """
    Calculate the combined loss with improved error handling and scaling.
    
    Args:
        outputs: Model outputs dictionary
        targets: Target outputs dictionary
        loss_weights: Dictionary of loss weights
        pad_idx: Index of padding token (to be ignored in sequence loss)
        label_smoothing: Label smoothing factor for cross-entropy
        
    Returns:
        total_loss: Combined weighted loss
        loss_info: Dictionary of individual loss components
    """
    # Sequence prediction loss (cross-entropy with label smoothing)
    seq_loss = nn.CrossEntropyLoss(
        ignore_index=pad_idx, 
        label_smoothing=label_smoothing
    )(
        outputs['sequence_probs'].reshape(-1, outputs['sequence_probs'].size(-1)),
        targets['target_sequence'].reshape(-1)
    )
    
    # CRITICAL FIX: Ensure values are strictly between 0 and 1 for BCE loss
    # For local PTM presence prediction
    safe_ptm_local_presence = torch.clamp(outputs['ptm_local_presence'], 1e-6, 1.0 - 1e-6)
    
    ptm_local_presence_loss = nn.BCELoss()(
        safe_ptm_local_presence.reshape(-1),
        targets['ptm_presence'].reshape(-1)
    )
    
    # Only compute offset loss for positions with PTMs
    ptm_mask = targets['ptm_presence'].float()
    
    # IMPROVEMENT: Use SmoothL1Loss for offset prediction to be more robust to outliers
    ptm_local_offset_loss = nn.SmoothL1Loss()(
        outputs['ptm_local_offset'] * ptm_mask,
        targets['ptm_offset'] * ptm_mask
    )
    
    # For global PTM presence prediction
    safe_ptm_global_presence = torch.clamp(outputs['ptm_global_presence'], 1e-6, 1.0 - 1e-6)
    
    ptm_global_presence_loss = nn.BCELoss()(
        safe_ptm_global_presence.reshape(-1),
        targets['global_ptm_presence'].reshape(-1)
    )
    
    # Global offset loss - MAJOR FIX: Scale down large offset values to prevent numerical issues
    global_presence_mask = targets['global_ptm_presence'].float()
    
    # Apply global PTM presence mask
    masked_global_offset_output = outputs['ptm_global_offset'] * global_presence_mask
    masked_global_offset_target = targets['global_ptm_offset'] * global_presence_mask
    
    # Calculate the maximum offset value for scaling
    max_offset = torch.max(torch.abs(masked_global_offset_target))
    
    # Apply scaling if values are very large (prevent numerical issues)
    if max_offset > 100.0:
        scale_factor = 100.0 / max_offset
        scaled_output = masked_global_offset_output * scale_factor
        scaled_target = masked_global_offset_target * scale_factor
        
        # Use SmoothL1Loss which is more robust to outliers
        ptm_global_offset_loss = nn.SmoothL1Loss()(scaled_output, scaled_target)
    else:
        # Use SmoothL1Loss which is more robust to outliers
        ptm_global_offset_loss = nn.SmoothL1Loss()(masked_global_offset_output, masked_global_offset_target)
    
    # Combine all losses with weights
    # IMPROVEMENT: Apply scaling to high-magnitude losses to balance components
    total_loss = (
        loss_weights['seq'] * seq_loss +
        loss_weights['ptm_local'] * (ptm_local_presence_loss + ptm_local_offset_loss) +
        loss_weights['ptm_global'] * (ptm_global_presence_loss + ptm_global_offset_loss)
    )
    
    # Return loss info for logging
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

def validate_tensors(outputs, targets):
    """
    Check tensor values and shapes for issues before loss calculation.
    
    Args:
        outputs: Model outputs dictionary
        targets: Target outputs dictionary
        
    Returns:
        valid: True if all checks pass, False otherwise
        message: Error message if validation fails
    """
    # Check for NaN or Inf values
    for key, tensor in outputs.items():
        if torch.isnan(tensor).any():
            return False, f"NaN values detected in outputs['{key}']"
        if torch.isinf(tensor).any():
            return False, f"Inf values detected in outputs['{key}']"
    
    # Check probability ranges
    for key in ['ptm_local_presence', 'ptm_global_presence']:
        if key in outputs:
            tensor = outputs[key]
            min_val = tensor.min().item()
            max_val = tensor.max().item()
            if min_val < 0 or max_val > 1:
                return False, f"Values out of range [0,1] in outputs['{key}']: min={min_val}, max={max_val}"
    
    # Check shape compatibilities
    shape_pairs = [
        ('sequence_probs', 'target_sequence'),
        ('ptm_local_presence', 'ptm_presence'),
        ('ptm_local_offset', 'ptm_offset'),
        ('ptm_global_presence', 'global_ptm_presence'),
        ('ptm_global_offset', 'global_ptm_offset')
    ]
    
    for out_key, target_key in shape_pairs:
        if out_key == 'sequence_probs':
            # Special case for sequence probs which has vocab dimension
            if outputs[out_key].size(0) != targets[target_key].size(0) or \
               outputs[out_key].size(1) != targets[target_key].size(1):
                return False, f"Shape mismatch: outputs['{out_key}'] {outputs[out_key].shape} vs targets['{target_key}'] {targets[target_key].shape}"
        else:
            # For all other tensors, check if shapes are broadcastable
            try:
                # This is a simple check - we're just making sure no errors occur
                # when adding the tensors together (which requires broadcastable shapes)
                _ = outputs[out_key] + torch.zeros_like(targets[target_key])
            except:
                return False, f"Shapes not broadcastable: outputs['{out_key}'] {outputs[out_key].shape} vs targets['{target_key}'] {targets[target_key].shape}"
    
    return True, "All checks passed"


def check_and_handle_nans(model, optimizer, console=None):
    """
    Check for NaN values in model parameters and gradients, and handle them by:
    1. Identifying modules with NaN values
    2. Clipping gradients where NaNs were found
    3. Reducing learning rate if NaNs are detected
    4. Returning information about where NaNs were found
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        console: Optional rich console for logging
        
    Returns:
        has_nans: True if NaNs were found, False otherwise
        nan_info: Dictionary with information about where NaNs were found
    """
    has_nans = False
    nan_info = {}
    
    # Check parameters
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            has_nans = True
            nan_info[f"param:{name}"] = True
            
            # Replace NaNs with zeros
            param.data = torch.nan_to_num(param.data, nan=0.0)
        
        # Check gradients if they exist
        if param.grad is not None and torch.isnan(param.grad).any():
            has_nans = True
            nan_info[f"grad:{name}"] = True
            
            # Replace NaN gradients with zeros
            param.grad = torch.nan_to_num(param.grad, nan=0.0)
    
    # If NaNs were found, reduce the learning rate
    if has_nans:
        if console:
            console.log("[yellow]NaN values detected! Reducing learning rate...[/yellow]")
        
        # Reduce learning rate by factor of 0.5
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
            if console:
                console.log(f"Learning rate reduced to {param_group['lr']:.6f}")
        
        # Apply more aggressive gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
    
    return has_nans, nan_info

def safe_forward_pass(model, features, teacher_forcing_ratio=1.0, temperature=1.0):
    """
    Execute model forward pass with NaN prevention and detection
    
    Args:
        model: The model
        features: Input features
        teacher_forcing_ratio: Teacher forcing ratio
        temperature: Temperature for softmax
        
    Returns:
        outputs: Model outputs or None if forward pass failed
        error_msg: Error message if forward pass failed, None otherwise
    """
    try:
        # Try regular forward pass
        outputs = model(features, teacher_forcing_ratio=teacher_forcing_ratio, temperature=temperature)
        
        # Check for NaN values in outputs
        for key, tensor in outputs.items():
            if torch.isnan(tensor).any():
                # Try again with more stable settings
                outputs = model(features, teacher_forcing_ratio=1.0, temperature=0.5)
                
                # Check again
                if torch.isnan(outputs[key]).any():
                    return None, f"NaN values detected in outputs['{key}']"
        
        return outputs, None
        
    except Exception as e:
        return None, f"Forward pass failed: {type(e).__name__}: {str(e)}"


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
    Train the model for one epoch with enhanced error handling and NaN prevention.
    
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
    
    # Keep track of successful and failed batches
    successful_batches = 0
    failed_batches = 0
    
    # Track consecutive NaN batches
    consecutive_nan_batches = 0
    
    # Get initial learning rate
    initial_lr = optimizer.param_groups[0]['lr']
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn()
    ) as progress:
        train_task = progress.add_task("Training", total=len(dataloader))
        
        for batch_idx, (features, targets) in enumerate(dataloader):
            try:
                # Move features to device
                features = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                          (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                          for k, v in features.items()}
                
                # Move targets to device
                targets = {k: v.to(device) for k, v in targets.items()}
                
                # Safe forward pass with NaN detection
                outputs, error_msg = safe_forward_pass(
                    model, 
                    features, 
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    temperature=temperature
                )
                
                if error_msg is not None:
                    progress.console.log(f"[yellow]Batch {batch_idx} forward pass issue: {error_msg}[/yellow]")
                    
                    # Try again with more conservative settings
                    progress.console.log("[yellow]Retrying with more conservative settings...[/yellow]")
                    outputs, error_msg = safe_forward_pass(
                        model, 
                        features, 
                        teacher_forcing_ratio=1.0,  # Full teacher forcing
                        temperature=0.5            # Lower temperature
                    )
                    
                    if error_msg is not None:
                        progress.console.log(f"[red]Retry failed: {error_msg}. Skipping batch.[/red]")
                        failed_batches += 1
                        consecutive_nan_batches += 1
                        
                        # If too many consecutive failures, reduce learning rate
                        if consecutive_nan_batches >= 5:
                            progress.console.log("[red]Too many consecutive failures. Reducing learning rate...[/red]")
                            for param_group in optimizer.param_groups:
                                param_group['lr'] *= 0.5
                            progress.console.log(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
                            consecutive_nan_batches = 0
                        
                        progress.update(train_task, advance=1)
                        continue
                
                # Reset consecutive NaN counter on success
                consecutive_nan_batches = 0
                
                # Apply safety clamping to probability outputs
                if 'ptm_local_presence' in outputs:
                    outputs['ptm_local_presence'] = torch.clamp(outputs['ptm_local_presence'], 1e-6, 1.0 - 1e-6)
                
                if 'ptm_global_presence' in outputs:
                    outputs['ptm_global_presence'] = torch.clamp(outputs['ptm_global_presence'], 1e-6, 1.0 - 1e-6)
                
                # Calculate loss
                loss, batch_loss_info = calculate_loss(
                    outputs, 
                    targets, 
                    loss_weights, 
                    pad_idx, 
                    label_smoothing
                )
                
                # Check if loss is NaN
                if torch.isnan(loss):
                    progress.console.log(f"[red]Loss is NaN for batch {batch_idx}. Skipping.[/red]")
                    failed_batches += 1
                    consecutive_nan_batches += 1
                    progress.update(train_task, advance=1)
                    continue
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Check for NaN in gradients and fix if needed
                has_nans, nan_info = check_and_handle_nans(model, optimizer, console=progress.console)
                
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                optimizer.step()
                
                # Only step the scheduler if no NaNs were detected
                if not has_nans:
                    scheduler.step()
                
                # Accumulate losses
                for k, v in batch_loss_info.items():
                    loss_info_sum[k] += v
                
                successful_batches += 1
                
                # Update progress
                progress.update(train_task, advance=1)
                
                # Log progress
                if batch_idx % log_interval == 0:
                    lr = optimizer.param_groups[0]['lr']
                    progress.console.log(f"Batch {batch_idx}/{len(dataloader)}, Loss: {batch_loss_info['total']:.4f}, LR: {lr:.6f}")
                
            except Exception as e:
                failed_batches += 1
                consecutive_nan_batches += 1
                progress.console.log(f"[red]Error in batch {batch_idx}: {type(e).__name__}: {str(e)}[/red]")
                
                # Print more detailed information for debugging
                import traceback
                traceback.print_exc()
                
                # Reduce learning rate if many errors
                if consecutive_nan_batches >= 5:
                    progress.console.log("[red]Too many consecutive failures. Reducing learning rate...[/red]")
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.5
                    progress.console.log(f"Learning rate reduced to {optimizer.param_groups[0]['lr']:.6f}")
                    consecutive_nan_batches = 0
                
                # Continue with next batch
                progress.update(train_task, advance=1)
                continue
    
    # Calculate averages (based on successful batches only)
    if successful_batches > 0:
        avg_loss_info = {k: v / successful_batches for k, v in loss_info_sum.items()}
        print(f"Epoch complete: {successful_batches} successful batches, {failed_batches} failed batches")
        
        # Check if learning rate was reduced substantially due to NaNs
        final_lr = optimizer.param_groups[0]['lr']
        if final_lr < initial_lr * 0.1:
            print(f"[yellow]WARNING: Learning rate was reduced significantly from {initial_lr:.6f} to {final_lr:.6f} due to instability[/yellow]")
        
        return avg_loss_info['total'], avg_loss_info
    else:
        print(f"[red]WARNING: All {failed_batches} batches failed in this epoch![/red]")
        # Return dummy values
        return float('inf'), {k: float('inf') for k in loss_info_sum}


def evaluate(
    model, 
    dataloader, 
    loss_weights, 
    device, 
    pad_idx=0, 
    label_smoothing=0.1,
    temperature=1.0,
    max_batches=None  # Optional parameter to limit evaluation batches
):
    """
    Evaluate the model on the given dataloader with improved error handling.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader to use for evaluation
        loss_weights: Dictionary of loss weights
        device: Device to use
        pad_idx: Index of padding token
        label_smoothing: Label smoothing factor
        temperature: Temperature for sequence generation
        max_batches: Maximum number of batches to evaluate (None = all)
        
    Returns:
        avg_loss: Average loss over the dataset
        loss_info: Dictionary of average loss components
    """
    model.eval()
    loss_info_sum = {
        'total': 0,
        'seq': 0,
        'ptm_local_presence': 0, 
        'ptm_local_offset': 0,
        'ptm_global_presence': 0,
        'ptm_global_offset': 0
    }
    
    # Track successful batches
    successful_batches = 0
    
    print("Starting evaluation...")
    with torch.no_grad():
        # Create progress bar for evaluation
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn()
        ) as progress:
            # Determine number of batches to evaluate
            total_batches = len(dataloader) if max_batches is None else min(max_batches, len(dataloader))
            eval_task = progress.add_task("Evaluating", total=total_batches)
            
            for batch_idx, (features, targets) in enumerate(dataloader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                    
                try:
                    # Move features to device
                    features = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                              (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                              for k, v in features.items()}
                    
                    # Move targets to device
                    targets = {k: v.to(device) for k, v in targets.items()}
                    
                    # Forward pass with safety measures
                    outputs = model(features, teacher_forcing_ratio=1.0, temperature=temperature)
                    
                    # Apply safety clamping
                    if 'ptm_local_presence' in outputs:
                        outputs['ptm_local_presence'] = torch.clamp(outputs['ptm_local_presence'], 1e-6, 1.0 - 1e-6)
                    
                    if 'ptm_global_presence' in outputs:
                        outputs['ptm_global_presence'] = torch.clamp(outputs['ptm_global_presence'], 1e-6, 1.0 - 1e-6)
                    
                    # Calculate loss with the fixed calculate_loss function
                    total_loss, batch_loss_info = calculate_loss(
                        outputs, 
                        targets, 
                        loss_weights, 
                        pad_idx, 
                        label_smoothing
                    )
                    
                    # Accumulate losses
                    for k, v in batch_loss_info.items():
                        loss_info_sum[k] += v
                    
                    successful_batches += 1
                    
                except Exception as e:
                    progress.console.log(f"[red]Error in evaluation batch {batch_idx}: {type(e).__name__}: {str(e)}[/red]")
                    
                    # Continue with next batch
                    continue
                finally:
                    # Always update progress
                    progress.update(eval_task, advance=1)
                    
                    # Free memory explicitly
                    if 'outputs' in locals():
                        del outputs
                    if 'features' in locals():
                        del features
                    if 'targets' in locals():
                        del targets
                    torch.cuda.empty_cache()
    
    print(f"Evaluation complete: {successful_batches} successful batches")
    
    # Calculate averages
    if successful_batches > 0:
        avg_loss_info = {k: v / successful_batches for k, v in loss_info_sum.items()}
        return avg_loss_info['total'], avg_loss_info
    else:
        print("[red]WARNING: All evaluation batches failed![/red]")
        return float('inf'), {k: float('inf') for k in loss_info_sum}

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

def analyze_predictions(console, model, features, targets, id_to_aa, max_samples=5, timeout=30):
    """
    Analyze model predictions with proper batch handling.
    
    Args:
        console: Rich console for output
        model: The model
        features: Batch features
        targets: Batch targets
        id_to_aa: ID to amino acid mapping
        max_samples: Maximum number of samples to display (default: 5)
        timeout: Maximum time in seconds for this function (default: 30)
    """
    import time
    start_time = time.time()
    model.eval()
    
    console.print("\n[bold]Sample Predictions:[/bold]")
    
    with torch.no_grad():
        try:
            # Get device
            device = next(model.parameters()).device
            
            # Process at most max_samples to avoid memory issues
            batch_size = min(max_samples, features['input_sequence'].size(0))
            
            # CRITICAL FIX: Create new feature dict with consistently sized tensors
            # Make sure ALL tensors have the same batch dimension
            small_features = {}
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    small_features[k] = v[:batch_size].to(device)
                elif isinstance(v, tuple) and len(v) == 2:
                    # Handle peaks tuple - both elements must have same first dimension
                    small_features[k] = (v[0][:batch_size].to(device), v[1][:batch_size].to(device))
                else:
                    small_features[k] = v
            
            small_targets = {k: v[:batch_size].to(device) for k, v in targets.items()}
            
            # Forward pass with low temperature (more peaked distribution)
            outputs = model(small_features, teacher_forcing_ratio=0.0, temperature=0.7)
            
            # Get predictions
            pred_sequence = outputs['sequence_probs'].argmax(dim=-1)  # [batch, seq_len]
            pred_ptm_presence = (outputs['ptm_local_presence'] > 0.5).float()  # [batch, seq_len]
            
            # Display results
            for i in range(batch_size):
                # Check timeout
                if time.time() - start_time > timeout:
                    console.print("[yellow]Analysis timeout reached. Stopping to prevent hanging.[/yellow]")
                    return
                
                # Get true sequence
                true_seq = decode_sequence(id_to_aa, small_targets['target_sequence'][i])
                
                # Get predicted sequence
                pred_seq = decode_sequence(id_to_aa, pred_sequence[i])
                
                # Get true PTMs
                true_ptm_pos = small_targets['ptm_presence'][i].nonzero().squeeze(-1).cpu().tolist()
                if not isinstance(true_ptm_pos, list):
                    true_ptm_pos = [true_ptm_pos]
                
                # Get predicted PTMs
                pred_ptm_pos = pred_ptm_presence[i].nonzero().squeeze(-1).cpu().tolist()
                if not isinstance(pred_ptm_pos, list):
                    pred_ptm_pos = [pred_ptm_pos]
                
                # Display
                console.print(f"Example {i+1}:")
                console.print(f"  True:    {true_seq}")
                console.print(f"  Pred:    {pred_seq}")
                console.print(f"  True PTM positions:  {true_ptm_pos}")
                console.print(f"  Pred PTM positions:  {pred_ptm_pos}")
                console.print("")
                
        except Exception as e:
            import traceback
            console.print(f"[red]Error during prediction analysis: {type(e).__name__}: {str(e)}[/red]")
            traceback.print_exc()
            
        finally:
            # Free memory explicitly
            if 'outputs' in locals():
                del outputs
            if 'small_features' in locals():
                del small_features
            if 'small_targets' in locals():
                del small_targets
            torch.cuda.empty_cache()

def predict_and_visualize(model, dataloader, id_to_aa, device, num_samples=5, timeout=60):
    """
    Make predictions and visualize results with proper batch handling.
    
    Args:
        model: The trained model
        dataloader: DataLoader to use
        id_to_aa: Dictionary mapping token IDs to amino acids
        device: Device to use for tensors
        num_samples: Number of samples to visualize
        timeout: Maximum time in seconds for this function
        
    Returns:
        Table with visualization results
    """
    import time
    start_time = time.time()
    model.eval()
    
    # Create table for results
    from rich.table import Table
    table = Table(title="Prediction Samples")
    table.add_column("Sample", justify="center")
    table.add_column("True Sequence", justify="left")
    table.add_column("Predicted Sequence", justify="left")
    table.add_column("True PTMs", justify="left")
    table.add_column("Predicted PTMs", justify="left")
    table.add_column("True Global PTM", justify="center")
    table.add_column("Predicted Global PTM", justify="center")
    
    try:
        with torch.no_grad():
            # Get a single batch from the dataloader
            features, targets = next(iter(dataloader))
            
            # Process at most num_samples to avoid memory issues
            batch_size = min(num_samples, features['input_sequence'].size(0))
            
            # CRITICAL FIX: Create new feature dict with consistently sized tensors
            # Make sure ALL tensors have the same batch dimension
            small_features = {}
            for k, v in features.items():
                if isinstance(v, torch.Tensor):
                    small_features[k] = v[:batch_size].to(device)
                elif isinstance(v, tuple) and len(v) == 2:
                    # Handle peaks tuple - both elements must have same first dimension
                    small_features[k] = (v[0][:batch_size].to(device), v[1][:batch_size].to(device))
                else:
                    small_features[k] = v
            
            small_targets = {k: v[:batch_size].to(device) for k, v in targets.items()}
            
            # Forward pass with lower temperature
            outputs = model(small_features, teacher_forcing_ratio=0.0, temperature=0.7)
            
            # Get predictions
            pred_sequence = outputs['sequence_probs'].argmax(dim=-1)  # [batch, seq_len]
            pred_ptm_presence = (outputs['ptm_local_presence'] > 0.5).float()  # [batch, seq_len]
            
            # Process batch
            for i in range(batch_size):
                # Check timeout
                if time.time() - start_time > timeout:
                    print("[yellow]Visualization timeout reached. Showing partial results.[/yellow]")
                    break
                
                # Get true sequence
                true_seq = decode_sequence(id_to_aa, small_targets['target_sequence'][i])
                
                # Get predicted sequence
                pred_seq = decode_sequence(id_to_aa, pred_sequence[i])
                
                # Get predicted PTMs
                pred_ptm_pos = pred_ptm_presence[i].nonzero().squeeze(-1).cpu().tolist()
                if not isinstance(pred_ptm_pos, list):
                    pred_ptm_pos = [pred_ptm_pos]
                pred_ptm_offsets = [f"{outputs['ptm_local_offset'][i, pos].item():.2f}" for pos in pred_ptm_pos]
                
                # Get true PTMs
                true_ptm_pos = small_targets['ptm_presence'][i].nonzero().squeeze(-1).cpu().tolist()
                if not isinstance(true_ptm_pos, list):
                    true_ptm_pos = [true_ptm_pos]
                true_ptm_offsets = [f"{small_targets['ptm_offset'][i, pos].item():.2f}" for pos in true_ptm_pos]
                
                # Get global PTM info
                pred_global_presence = outputs['ptm_global_presence'][i].item() > 0.5
                pred_global_offset = outputs['ptm_global_offset'][i].item()
                true_global_presence = small_targets['global_ptm_presence'][i].item() > 0.5
                true_global_offset = small_targets['global_ptm_offset'][i].item()
                
                # Format data for table
                true_ptms = ", ".join([f"Pos {pos}: {offset}" for pos, offset in zip(true_ptm_pos, true_ptm_offsets)])
                pred_ptms = ", ".join([f"Pos {pos}: {offset}" for pos, offset in zip(pred_ptm_pos, pred_ptm_offsets)])
                
                true_global = f"{'Yes' if true_global_presence else 'No'} ({true_global_offset:.2f})"
                pred_global = f"{'Yes' if pred_global_presence else 'No'} ({pred_global_offset:.2f})"
                
                # Add to table
                table.add_row(
                    str(i+1),
                    true_seq,
                    pred_seq,
                    true_ptms if true_ptms else "None",
                    pred_ptms if pred_ptms else "None",
                    true_global,
                    pred_global
                )
                
    except Exception as e:
        import traceback
        print(f"[red]Error during visualization: {type(e).__name__}: {str(e)}[/red]")
        traceback.print_exc()
        
    finally:
        # Free memory explicitly
        if 'outputs' in locals():
            del outputs
        if 'small_features' in locals():
            del small_features
        if 'small_targets' in locals():
            del small_targets
        torch.cuda.empty_cache()
    
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
    """Main training function with improved error handling."""
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
    try:
        train_loader, val_loader = create_train_test_dataloaders(
            args.msp_file,
            train_ratio=0.8,
            max_peaks=args.max_peaks,
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            random_seed=args.seed
        )
    except Exception as e:
        console.log(f"[red]Error creating dataloaders: {type(e).__name__}: {str(e)}[/red]")
        raise
    
    # Get vocabulary size from the dataloader
    vocab_size = len(train_loader.dataset.aa_to_id)
    id_to_aa = train_loader.dataset.id_to_aa
    pad_idx = train_loader.dataset.aa_to_id['_']
    
    console.log(f"Vocabulary size: {vocab_size}")
    
    # Create model
    console.log("Creating model...")
    try:
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
    except Exception as e:
        console.log(f"[red]Error creating model: {type(e).__name__}: {str(e)}[/red]")
        raise
    
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
        try:
            start_epoch = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
        except Exception as e:
            console.log(f"[yellow]Error loading checkpoint: {type(e).__name__}: {str(e)}. Starting from scratch.[/yellow]")
            start_epoch = 0
    
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
        try:
            train_loss, train_loss_info = train_one_epoch(
                model, train_loader, optimizer, scheduler, loss_weights, device,
                pad_idx=pad_idx, 
                log_interval=args.log_interval,
                max_grad_norm=args.max_grad_norm,
                label_smoothing=args.label_smoothing,
                teacher_forcing_ratio=teacher_forcing_ratio,
                temperature=args.temperature
            )
            console.log(f"Train Loss: {train_loss:.4f}")
        except Exception as e:
            console.log(f"[red]Error during training epoch {epoch+1}: {type(e).__name__}: {str(e)}[/red]")
            train_loss = float('inf')
            train_loss_info = {k: float('inf') for k in ['seq', 'ptm_local_presence', 'ptm_local_offset', 
                                                         'ptm_global_presence', 'ptm_global_offset']}
        
        # Free memory after training
        torch.cuda.empty_cache()
        
        # Evaluate on validation set (with limited batches to prevent hangs)
        try:
            val_loss, val_loss_info = evaluate(
                model, val_loader, loss_weights, device, 
                pad_idx=pad_idx,
                label_smoothing=args.label_smoothing,
                temperature=args.temperature,
                max_batches=min(100, len(val_loader))  # Limit eval batches to prevent hanging
            )
            console.log(f"Val Loss: {val_loss:.4f}")
        except Exception as e:
            console.log(f"[red]Error during validation for epoch {epoch+1}: {type(e).__name__}: {str(e)}[/red]")
            val_loss = float('inf')
            val_loss_info = {k: float('inf') for k in ['seq', 'ptm_local_presence', 'ptm_local_offset', 
                                                       'ptm_global_presence', 'ptm_global_offset']}
        
        # Free memory after evaluation
        torch.cuda.empty_cache()
        
        # Log metrics
        console.log("Loss Components:")
        for k, v in train_loss_info.items():
            if k == 'total':
                continue
            console.log(f"  Train {k}: {v:.4f}, Val {k}: {val_loss_info.get(k, float('inf')):.4f}")
        
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
        
        # Analyze predictions on a small batch with timeout protection
        try:
            # Get a small batch for analysis
            eval_features, eval_targets = next(iter(val_loader))
            analyze_predictions(console, model, eval_features, eval_targets, id_to_aa, max_samples=5, timeout=30)
            torch.cuda.empty_cache()  # Free memory after analysis
        except Exception as e:
            console.log(f"[yellow]Error during prediction analysis: {type(e).__name__}: {str(e)}[/yellow]")
        
        # Save checkpoint with error handling
        if (epoch + 1) % args.save_interval == 0:
            try:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {'train': train_loss_info, 'val': val_loss_info},
                    os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                )
            except Exception as e:
                console.log(f"[yellow]Error saving checkpoint for epoch {epoch+1}: {type(e).__name__}: {str(e)}[/yellow]")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            try:
                save_checkpoint(
                    model, optimizer, scheduler, epoch,
                    {'train': train_loss_info, 'val': val_loss_info},
                    os.path.join(args.output_dir, 'best_model.pt')
                )
                console.log("[green]New best model saved![/green]")
            except Exception as e:
                console.log(f"[yellow]Error saving best model: {type(e).__name__}: {str(e)}[/yellow]")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                console.log(f"[yellow]Early stopping after {patience_counter} epochs without improvement[/yellow]")
                break
        
        # Save latest checkpoint
        try:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                {'train': train_loss_info, 'val': val_loss_info},
                checkpoint_path
            )
        except Exception as e:
            console.log(f"[yellow]Error saving latest checkpoint: {type(e).__name__}: {str(e)}[/yellow]")
        
        # Show some predictions with visualization table
        if (epoch + 1) % args.save_interval == 0:
            try:
                with torch.no_grad():
                    table = predict_and_visualize(model, val_loader, id_to_aa, device, num_samples=5, timeout=60)
                    console.print(table)
                torch.cuda.empty_cache()  # Free memory after visualization
            except Exception as e:
                console.log(f"[yellow]Error during prediction visualization: {type(e).__name__}: {str(e)}[/yellow]")
    
    console.log("[green]Training complete![/green]")
    
    # Final evaluation with limited batches
    console.log("Final evaluation...")
    try:
        test_loss, test_loss_info = evaluate(
            model, val_loader, loss_weights, device, 
            pad_idx=pad_idx,
            label_smoothing=args.label_smoothing,
            max_batches=min(100, len(val_loader))  # Limit eval batches
        )
        console.log(f"Test Loss: {test_loss:.4f}")
    except Exception as e:
        console.log(f"[red]Error during final evaluation: {type(e).__name__}: {str(e)}[/red]")
    
    # Visualize final predictions
    try:
        with torch.no_grad():
            table = predict_and_visualize(model, val_loader, id_to_aa, device, num_samples=5, timeout=60)
            console.print(table)
    except Exception as e:
        console.log(f"[yellow]Error during final visualization: {type(e).__name__}: {str(e)}[/yellow]")
    
    # Save final model
    try:
        save_checkpoint(
            model, optimizer, scheduler, args.epochs - 1,
            {'val': test_loss_info if 'test_loss_info' in locals() else {}},
            os.path.join(args.output_dir, 'final_model.pt')
        )
    except Exception as e:
        console.log(f"[yellow]Error saving final model: {type(e).__name__}: {str(e)}[/yellow]")
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    args = parse_args()
    train_model(args)