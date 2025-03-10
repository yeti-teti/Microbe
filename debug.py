import os
import torch
import numpy as np
import argparse
from data_loader import create_train_test_dataloaders
from model import Spec2PepWithPTM

def debug_model(args):
    """Debug model by running one forward pass with detailed tensor checks"""
    print("========== Starting Model Debug ==========")
    
    # Set environment variables for better error reporting
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    
    # Create dataloaders with smaller batch size for debugging
    print("Loading data...")
    train_loader, val_loader = create_train_test_dataloaders(
        args.msp_file,
        train_ratio=0.8,
        max_peaks=args.max_peaks,
        max_seq_len=args.max_seq_len,
        batch_size=2,  # Small batch size for debugging
        random_seed=args.seed
    )
    
    # Get vocabulary size
    vocab_size = len(train_loader.dataset.aa_to_id)
    id_to_aa = train_loader.dataset.id_to_aa
    pad_idx = train_loader.dataset.aa_to_id['_']
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Create model
    print("Creating model...")
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
    
    # Set to eval mode for consistent behavior
    model.eval()
    
    # Get a single batch
    print("Getting batch...")
    features, targets = next(iter(train_loader))
    
    # Print shapes
    print("\n===== Batch Shapes =====")
    print(f"Batch size: {features['input_sequence'].shape[0]}")
    
    # Print features
    for key, value in features.items():
        if isinstance(value, torch.Tensor):
            print(f"Feature '{key}': shape={value.shape}, dtype={value.dtype}")
        elif isinstance(value, tuple) and len(value) == 2:
            print(f"Feature '{key}': tuple of shapes {value[0].shape} and {value[1].shape}")
        else:
            print(f"Feature '{key}': {type(value)}")
    
    # Print targets
    for key, value in targets.items():
        print(f"Target '{key}': shape={value.shape}, dtype={value.dtype}")
    
    # Move data to device
    print("\n===== Moving data to device =====")
    features_device = {k: v.to(device) if isinstance(v, torch.Tensor) else 
                     (v[0].to(device), v[1].to(device)) if isinstance(v, tuple) else v 
                     for k, v in features.items()}
    
    targets_device = {k: v.to(device) for k, v in targets.items()}
    
    # Forward pass
    print("\n===== Trying forward pass =====")
    try:
        with torch.no_grad():
            outputs = model(features_device, teacher_forcing_ratio=1.0, temperature=1.0)
        
        print("Forward pass successful!")
        
        # Check outputs
        print("\n===== Model Outputs =====")
        for key, value in outputs.items():
            print(f"Output '{key}': shape={value.shape}, dtype={value.dtype}")
            # Check for NaN or Inf values
            if torch.isnan(value).any() or torch.isinf(value).any():
                print(f"WARNING: {key} contains NaN or Inf values!")
            
            # Check range for probabilities
            if 'prob' in key or 'presence' in key:
                min_val = value.min().item()
                max_val = value.max().item()
                print(f"  Range: [{min_val}, {max_val}]")
                if min_val < 0 or max_val > 1:
                    print(f"  WARNING: Values outside [0, 1] range!")
        
        # Try calculating loss
        print("\n===== Trying loss calculation =====")
        
        # Sequence prediction loss
        seq_loss = torch.nn.CrossEntropyLoss(
            ignore_index=pad_idx, 
            label_smoothing=0.1
        )(
            outputs['sequence_probs'].reshape(-1, outputs['sequence_probs'].size(-1)),
            targets_device['target_sequence'].reshape(-1)
        )
        print(f"Sequence loss: {seq_loss.item()}")
        
        # Local PTM prediction loss
        ptm_local_presence_loss = torch.nn.BCELoss()(
            outputs['ptm_local_presence'].reshape(-1),
            targets_device['ptm_presence'].reshape(-1)
        )
        print(f"PTM local presence loss: {ptm_local_presence_loss.item()}")
        
        # Local PTM offset loss
        ptm_mask = targets_device['ptm_presence'].float()
        ptm_local_offset_loss = torch.nn.MSELoss()(
            outputs['ptm_local_offset'] * ptm_mask,
            targets_device['ptm_offset'] * ptm_mask
        )
        print(f"PTM local offset loss: {ptm_local_offset_loss.item()}")
        
        # Global PTM presence loss
        global_ptm_presence_output = outputs['ptm_global_presence']
        global_ptm_presence_target = targets_device['global_ptm_presence']
        
        # Print detailed shapes for debugging
        print(f"\nGlobal presence shapes: output={global_ptm_presence_output.shape}, target={global_ptm_presence_target.shape}")
        
        # Handle dimension mismatches
        if global_ptm_presence_output.dim() != global_ptm_presence_target.dim():
            print(f"Dimensions don't match: output={global_ptm_presence_output.dim()}, target={global_ptm_presence_target.dim()}")
            if global_ptm_presence_output.dim() > global_ptm_presence_target.dim():
                global_ptm_presence_output = global_ptm_presence_output.squeeze()
            else:
                global_ptm_presence_target = global_ptm_presence_target.squeeze()
            print(f"After squeezing: output={global_ptm_presence_output.shape}, target={global_ptm_presence_target.shape}")
        
        # Ensure both have the same shape
        if global_ptm_presence_output.shape != global_ptm_presence_target.shape:
            print(f"Shapes don't match after squeezing, trying to reshape")
            try:
                global_ptm_presence_output = global_ptm_presence_output.reshape(global_ptm_presence_target.shape)
                print(f"After reshaping: output={global_ptm_presence_output.shape}, target={global_ptm_presence_target.shape}")
            except:
                print(f"Reshape failed, will try broadcasting in BCE loss")
        
        ptm_global_presence_loss = torch.nn.BCELoss()(
            global_ptm_presence_output,
            global_ptm_presence_target
        )
        print(f"PTM global presence loss: {ptm_global_presence_loss.item()}")
        
        # Global PTM offset loss
        global_ptm_offset_output = outputs['ptm_global_offset']
        global_ptm_offset_target = targets_device['global_ptm_offset']
        
        print(f"\nGlobal offset shapes: output={global_ptm_offset_output.shape}, target={global_ptm_offset_target.shape}")
        
        # Handle dimension mismatches
        if global_ptm_offset_output.dim() != global_ptm_offset_target.dim():
            print(f"Dimensions don't match: output={global_ptm_offset_output.dim()}, target={global_ptm_offset_target.dim()}")
            if global_ptm_offset_output.dim() > global_ptm_offset_target.dim():
                global_ptm_offset_output = global_ptm_offset_output.squeeze()
            else:
                global_ptm_offset_target = global_ptm_offset_target.squeeze()
            print(f"After squeezing: output={global_ptm_offset_output.shape}, target={global_ptm_offset_target.shape}")
        
        # Ensure both have the same shape
        if global_ptm_offset_output.shape != global_ptm_offset_target.shape:
            print(f"Shapes don't match after squeezing, trying to reshape")
            try:
                global_ptm_offset_output = global_ptm_offset_output.reshape(global_ptm_offset_target.shape)
                print(f"After reshaping: output={global_ptm_offset_output.shape}, target={global_ptm_offset_target.shape}")
            except:
                print(f"Reshape failed, will try simple element-wise operations")
        
        # Global presence mask
        global_presence_mask = targets_device['global_ptm_presence'].float()
        if global_presence_mask.dim() > 1:
            global_presence_mask = global_presence_mask.squeeze()
        
        print(f"Global presence mask shape: {global_presence_mask.shape}")
        
        # Try masking
        try:
            masked_global_offset_output = global_ptm_offset_output * global_presence_mask
            masked_global_offset_target = global_ptm_offset_target * global_presence_mask
            
            print(f"Masked shapes: output={masked_global_offset_output.shape}, target={masked_global_offset_target.shape}")
            
            ptm_global_offset_loss = torch.nn.MSELoss()(
                masked_global_offset_output,
                masked_global_offset_target
            )
            print(f"PTM global offset loss: {ptm_global_offset_loss.item()}")
            
            # Combined loss
            total_loss = (
                seq_loss +
                0.5 * (ptm_local_presence_loss + ptm_local_offset_loss) +
                0.5 * (ptm_global_presence_loss + ptm_global_offset_loss)
            )
            print(f"Total loss: {total_loss.item()}")
            
            print("\nLoss calculation successful!")
            
        except Exception as e:
            print(f"Error in global offset loss calculation: {e}")
            print("Attempting simplified calculation...")
            
            # Try a simpler approach
            global_ptm_offset_output_flat = global_ptm_offset_output.reshape(-1)
            global_ptm_offset_target_flat = global_ptm_offset_target.reshape(-1)
            print(f"Flattened shapes: output={global_ptm_offset_output_flat.shape}, target={global_ptm_offset_target_flat.shape}")
            
            global_offset_loss = torch.nn.MSELoss()(
                global_ptm_offset_output_flat[:1],  # Just use first element for simplicity
                global_ptm_offset_target_flat[:1]
            )
            print(f"Simplified global offset loss: {global_offset_loss.item()}")
    
    except Exception as e:
        print(f"\n===== ERROR =====")
        print(f"Exception: {type(e).__name__}")
        print(f"Message: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
    
    print("\n========== Debug Complete ==========")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Debug Spec2Pep model with PTM prediction")
    
    # Add necessary arguments
    parser.add_argument("--msp_file", type=str, required=True, help="Path to MSP file")
    parser.add_argument("--max_peaks", type=int, default=200, help="Maximum number of peaks to use")
    parser.add_argument("--max_seq_len", type=int, default=50, help="Maximum sequence length")
    parser.add_argument("--d_model", type=int, default=512, help="Model dimension")
    parser.add_argument("--d_ff", type=int, default=2048, help="Feed-forward dimension")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    debug_model(args)