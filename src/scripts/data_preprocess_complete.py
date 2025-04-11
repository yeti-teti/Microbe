import os
import re
import csv

import numpy as np


# MSP Parsing and SPectrum Structuring
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

# Intensity Normalization and Transformation
def preprocess_intensities(spec):
    total = spec["intensity"].sum()
    if total > 0:
        spec["intensity"] = spec["intensity"] / total
        spec["intensity"] = np.sqrt(spec["intensity"])
    else:
        spec["intensity"] = np.zeros_like(spec["intensity"])  # Or skip this spectrum

# Function to apply modifications to sequence
def apply_modifications(sequence, mods_str, ptm_dict):
    
    """Apply modifications to a peptide sequence based on the Mods string."""
    if mods_str == "0":
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
def msp_to_mgf(msp_filename, mgf_filename):
    """Convert an MSP file to an annotated MGF file for Casanovo training."""
    ptms = {
        "CAM": 57.02146,      # Carbamidomethylation on Cysteine
        "Oxidation": 15.994915,      # Oxidation (commonly on Methionine)
        "Acetyl": 42.010565,  # Acetylation (N-term or Lysine) 
    }
    
    out = open(mgf_filename, "w")
    spectrum_index = 0
    for spec in parse_msp(msp_filename):
        preprocess_intensities(spec)
        if "sequence" not in spec:
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
        check_mod = int(mods[0])
        if(check_mod != 0):
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
    
    out.close()
    print(f"Converted MSP '{msp_filename}' to MGF '{mgf_filename}'.")
            

if __name__ == "__main__":
    print("Converting the data...")
    msp_file = "human_hcd_tryp_best.msp"
    mgf_file = "human_hcd_tryp_best.mgf"
    if not os.path.exists(mgf_file):
        print("Converting to MGF format")
        msp_to_mgf(msp_file, mgf_file)

    print("Data converted...")