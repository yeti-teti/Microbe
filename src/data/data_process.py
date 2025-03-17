import random

def split_mgf_file(input_file, train_file, val_file, val_ratio=0.2, seed=42):
    random.seed(seed)
    
    # Read the input file
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Split into individual spectra
    spectra = []
    current_spectrum = []
    for line in content.split('\n'):
        current_spectrum.append(line)
        if line.strip() == "END IONS":
            spectra.append('\n'.join(current_spectrum))
            current_spectrum = []
    
    # Shuffle and split
    random.shuffle(spectra)
    split_idx = int(len(spectra) * (1 - val_ratio))
    train_spectra = spectra[:split_idx]
    val_spectra = spectra[split_idx:]
    
    # Write to output files
    with open(train_file, 'w') as f:
        f.write('\n'.join(train_spectra))
    
    with open(val_file, 'w') as f:
        f.write('\n'.join(val_spectra))
    
    print(f"Split {len(spectra)} spectra into {len(train_spectra)} training and {len(val_spectra)} validation spectra")

# Example usage
split_mgf_file(
    "../datasets/human_hcd_tryp_best.mgf", 
    "../datasets/train_set.mgf", 
    "../datasets/val_set.mgf", 
    val_ratio=0.2
)