import os
import re
import csv
import numpy as np

# MSP Parsing and Spectrum Structuring
def parse_msp(filename, limit=None):
    """
    Parse an MSP spectral library file and yield spectra as dictionaries.
    
    Args:
        filename: Path to the MSP file
        limit: Maximum number of spectra to parse (None for all)
    """
    spectrum = {}
    peaks_mz, peaks_int, peaks_ann = [], [], []
    spectrum_count = 0
    
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
                    spectrum_count += 1
                    
                    # Check if we've reached the limit
                    if limit is not None and spectrum_count >= limit:
                        return
                
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
    
    # Handle the last spectrum
    if spectrum and (limit is None or spectrum_count < limit):
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

# Intensity Normalization and Transformation
def preprocess_intensities(spec):
    """Apply TIC normalization and square-root transform to intensities."""
    total = spec["intensity"].sum()
    if total > 0 and not np.isnan(total) and not np.isinf(total):
        spec["intensity"] = spec["intensity"] / total
        spec["intensity"] = np.sqrt(spec["intensity"])
    else:
        # Set intensities to zero if total is invalid to avoid NaN
        spec["intensity"] = np.zeros_like(spec["intensity"])
        print(f"Warning: Spectrum '{spec.get('Name', 'unknown')}' has invalid intensity sum ({total}), setting intensities to zero.")

# Function to apply modifications to sequence
def apply_modifications(sequence, mods_str, ptm_dict):
    """Apply modifications to a peptide sequence based on the Mods string."""
    if mods_str == "0" or not mods_str:
        return sequence 
    
    # Extract the number of modifications and the modification details
    num_mods = mods_str.split("(")[0]
    if num_mods == "0":
        return sequence
    
    # Parse the modifications
    pattern = r'\(([^)]+)\)'
    matches = re.findall(pattern, mods_str)
    
    # Transform sequence to list for easier modification
    seq_list = list(sequence)
    
    # Sort modifications by position in descending order to avoid index shifts
    all_ptms = [match.split(',') for match in matches]
    all_ptms.sort(key=lambda x: int(x[0]), reverse=True)
    
    for ptm in all_ptms:
        pos = int(ptm[0])
        mod_type = ptm[2]
        if mod_type in ptm_dict:
            mass_shift = ptm_dict[mod_type]
            # Insert the mass shift after the amino acid
            seq_list[pos] = seq_list[pos] + f"+{mass_shift}"
    
    return "".join(seq_list)

# MSP to MGF Converter
def msp_to_mgf(msp_filename, mgf_filename, limit=None):
    """
    Convert an MSP file to an annotated MGF file for Casanovo training.
    
    Args:
        msp_filename: Path to the input MSP file
        mgf_filename: Path to the output MGF file
        limit: Maximum number of spectra to process (None for all)
    """
    ptms = {
        "CAM": 57.02146,      # Carbamidomethylation on Cysteine
        "Oxidation": 15.994915,      # Oxidation (commonly on Methionine)
        "Acetyl": 42.010565,  # Acetylation (N-term or Lysine) 
    }
    
    out = open(mgf_filename, "w")
    spectrum_index = 0
    
    print(f"Converting up to {limit if limit else 'all'} spectra from MSP to MGF...")
    
    for spec in parse_msp(msp_filename, limit=limit):
        # Validate data for NaN or infinity
        if np.any(np.isnan(spec["mz"])) or np.any(np.isinf(spec["mz"])) or \
           np.any(np.isnan(spec["intensity"])) or np.any(np.isinf(spec["intensity"])):
            print(f"Skipping spectrum '{spec.get('Name', 'unknown')}' due to NaN or infinite values in mz or intensity.")
            continue
        
        preprocess_intensities(spec)
        
        # Check if intensities are valid after preprocessing
        if np.any(np.isnan(spec["intensity"])) or np.any(np.isinf(spec["intensity"])):
            print(f"Skipping spectrum '{spec.get('Name', 'unknown')}' due to invalid intensities after preprocessing.")
            continue
        
        if "sequence" not in spec:
            print(f"Skipping spectrum '{spec.get('Name', 'unknown')}' due to missing sequence.")
            continue
        
        # Get the pepmass (precursor m/z)
        pepmass = spec.get("Parent", "0.0")
        
        # Get charge
        charge = spec.get("Charge", "2")
        if not charge.endswith("+"):
            charge = charge + "+"
        
        # Apply modifications to sequence
        seq = spec["sequence"]
        mods = spec.get("mods", "0")
        check_mod = int(mods[0]) if mods else 0
        if check_mod != 0:
            modified_seq = apply_modifications(seq, mods, ptms)
        else:
            modified_seq = seq
        
        out.write("BEGIN IONS\n")
        out.write(f"TITLE=Spec_{spectrum_index}\n")
        out.write(f"PEPMASS={pepmass}\n")
        out.write(f"CHARGE={charge}\n")
        out.write(f"SEQ={modified_seq}\n")
        
        for mz, inten in zip(spec["mz"], spec["intensity"]):
            out.write(f"{mz:.4f} {inten:.4f}\n")
        
        out.write("END IONS\n\n")
        spectrum_index += 1
        
        # Show progress every 1000 spectra
        if spectrum_index % 1000 == 0:
            print(f"Processed {spectrum_index} spectra...")
    
    out.close()
    print(f"Converted {spectrum_index} spectra from '{msp_filename}' to '{mgf_filename}'.")

if __name__ == "__main__":
    print("Starting data conversion...")
    msp_file = "human_hcd_tryp_best.msp"
    mgf_file = "human_hcd_tryp_best.mgf"
    
    # Process up to 100,000 spectra
    limit = 100000
    
    if not os.path.exists(mgf_file):
        print(f"Converting to MGF format (limit: {limit} spectra)")
        msp_to_mgf(msp_file, mgf_file, limit=limit)
    else:
        print(f"MGF file '{mgf_file}' already exists. Delete it to reconvert.")

    print("Data conversion completed.")