import pandas as pd

# Read the mzTab file
with open("predictions.mztab", "r") as f:
    lines = f.readlines()

# Find the PSH line index
psh_index = next(i for i, line in enumerate(lines) if line.startswith("PSH"))

# Extract the header (excluding "PSH")
header = lines[psh_index].strip().split("\t")[1:]  # 20 columns

# Process PSM lines
psm_lines = []
for line in lines[psh_index + 1:]:
    if line.startswith("PSM"):
        parts = line.strip().split("\t")[1:]  # Exclude "PSM"
        if len(parts) == len(header) - 1:  # 19 columns (missing total_ptm_mass_shift)
            parts.append("0.0")  # Add default total_ptm_mass_shift
            psm_lines.append(parts)
        elif len(parts) == len(header):  # 20 columns
            psm_lines.append(parts)
        else:
            print(f"Skipping line with {len(parts)} columns: {line.strip()}")

# Check if there are any valid PSM lines
if not psm_lines:
    raise ValueError("No valid PSM lines found")

# Create DataFrame
mztab_df = pd.DataFrame(psm_lines, columns=header)

# Convert total_ptm_mass_shift to float
mztab_df["total_ptm_mass_shift"] = pd.to_numeric(mztab_df["total_ptm_mass_shift"], errors='coerce').fillna(0.0)

# Output each peptide and its PTM mass shift
print("Peptide sequences and their total PTM mass shifts:")
for index, row in mztab_df.iterrows():
    print(f"Peptide: {row['sequence']}, Total PTM Mass Shift: {row['total_ptm_mass_shift']} Da")

# Calculate and output the total PTM value for the protein
total_ptm_value = mztab_df["total_ptm_mass_shift"].sum()
print(f"\nTotal PTM value for the protein: {total_ptm_value} Da")