#!/usr/bin/env python3
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def read_mztab(filename="predictions.mztab"):
    """
    Read an mzTab file (with '*' as delimiter) and return a pandas DataFrame.
    The file must have a header line that starts with "PSH" and PSM lines that start with "PSM".
    """
    with open(filename, "r") as f:
        lines = f.readlines()

    # Find header row (line starting with "PSH")
    psh_index = next((i for i, line in enumerate(lines) if line.startswith("PSH")), None)
    if psh_index is None:
        raise ValueError("No PSH header found in the mzTab file.")

    header = lines[psh_index].strip().split("*")[1:]  # Remove the "PSH" token
    psm_lines = []
    for line in lines[psh_index:]:
        if line.startswith("PSM"):
            parts = line.strip().split("*")[1:]
            # If missing the total_ptm_mass_shift column, append default "0.0"
            if len(parts) == len(header) - 1:
                parts.append("0.0")
                psm_lines.append(parts)
            elif len(parts) == len(header):
                psm_lines.append(parts)
            else:
                print(f"Skipping line with {len(parts)} columns: {line.strip()}")
    if not psm_lines:
        raise ValueError("No valid PSM lines found in the mzTab file.")
    df = pd.DataFrame(psm_lines, columns=header)
    # Convert desired columns to numeric
    numeric_cols = ["total_ptm_mass_shift", "calc_mass_to_charge", "exp_mass_to_charge", "search_engine_score[1]", "charge"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def compute_peptide_mass(row):
    """
    Compute the neutral peptide mass using:
        (charge * calc_mass_to_charge) - (charge * proton_mass)
    """
    proton_mass = 1.007825
    return row["charge"] * row["calc_mass_to_charge"] - row["charge"] * proton_mass

def extract_mods(mod_field):
    """
    Given a modifications field string (e.g., 
      "5-[CHEMMOD, CHEMMOD:+57.02146, , ],8-[CHEMMOD, CHEMMOD:+15.99491, , ]"),
    extract each modification identifier.
    
    This function uses a regex to match each token in the form: 
         "<position>-[ ... ]"
    and then splits the content in the square brackets by commas.
    It returns a list of modification IDs (for this example, the second element 
    is assumed to be the mod ID, e.g. "CHEMMOD:+57.02146").
    """
    mod_field = mod_field.strip()
    if mod_field.lower() == "null":
        return []
    mods = []
    # Pattern to capture tokens of the form: <number>-[ ... ]
    pattern = r"\d+-\[(.*?)\]"
    tokens = re.findall(pattern, mod_field)
    for token in tokens:
        # Token may be like: "CHEMMOD, CHEMMOD:+57.02146, ,"
        parts = [p.strip() for p in token.split(",") if p.strip()]
        if len(parts) >= 2:
            mods.append(parts[1])
        else:
            mods.append(token)
    return mods

def save_plot(fig, filepath):
    """Save the given figure to the specified file and close the figure."""
    fig.savefig(filepath, bbox_inches="tight")
    plt.close(fig)

def main():
    output_dir = "visual"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the mzTab file
    df = read_mztab("predictions.mztab")
    
    # Compute additional columns
    df["peptide_mass"] = df.apply(compute_peptide_mass, axis=1)
    
    # ------------------ 1. Histograms ------------------
    # Histogram: Distribution of Total PTM Mass Shifts
    fig, ax = plt.subplots()
    ax.hist(df["total_ptm_mass_shift"], bins=50, edgecolor="black")
    ax.set_xlabel("Total PTM Mass Shift (Da)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Total PTM Mass Shifts")
    save_plot(fig, os.path.join(output_dir, "hist_total_ptm_mass_shifts.png"))
    
    # Histogram: Distribution of Peptide Masses
    fig, ax = plt.subplots()
    ax.hist(df["peptide_mass"], bins=50, edgecolor="black")
    ax.set_xlabel("Peptide Mass (Da)")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of Peptide Masses")
    save_plot(fig, os.path.join(output_dir, "hist_peptide_masses.png"))
    
    # Histogram: Distribution of PSM Scores
    fig, ax = plt.subplots()
    ax.hist(df["search_engine_score[1]"], bins=50, edgecolor="black")
    ax.set_xlabel("PSM Score")
    ax.set_ylabel("Frequency")
    ax.set_title("Histogram of PSM Scores")
    save_plot(fig, os.path.join(output_dir, "hist_psm_scores.png"))
    
    # ------------------ 2. Bar Chart: Top 20 Modification Frequencies ------------------
    mod_series = df["modifications"].apply(extract_mods)
    # Flatten the list of modifications
    all_mods = [mod for mod_list in mod_series for mod in mod_list]
    mod_counts = Counter(all_mods)
    top20 = mod_counts.most_common(20)
    if top20:
        mods, counts = zip(*top20)
    else:
        mods, counts = [], []
    fig, ax = plt.subplots()
    ax.bar(mods, counts, edgecolor="black")
    ax.set_xlabel("Modification")
    ax.set_ylabel("Frequency")
    ax.set_title("Top 20 Modifications by Frequency")
    plt.xticks(rotation=45, ha="right")
    save_plot(fig, os.path.join(output_dir, "bar_top20_modification_frequency.png"))
    
    # ------------------ 3. Scatter Plot: Experimental vs. Calculated m/z ------------------
    fig, ax = plt.subplots()
    ax.scatter(df["exp_mass_to_charge"], df["calc_mass_to_charge"], alpha=0.7)
    ax.set_xlabel("Experimental m/z")
    ax.set_ylabel("Calculated m/z")
    ax.set_title("Experimental vs. Calculated m/z")
    # Draw a diagonal reference line (x=y)
    lim_low = min(df["exp_mass_to_charge"].min(), df["calc_mass_to_charge"].min())
    lim_high = max(df["exp_mass_to_charge"].max(), df["calc_mass_to_charge"].max())
    ax.plot([lim_low, lim_high], [lim_low, lim_high], "r--")
    save_plot(fig, os.path.join(output_dir, "scatter_exp_vs_calc_mz.png"))
    
    # ------------------ 4. Box Plot: Total PTM Mass Shift by Charge State ------------------
    fig, ax = plt.subplots()
    sns.boxplot(x="charge", y="total_ptm_mass_shift", data=df, ax=ax)
    ax.set_xlabel("Charge State")
    ax.set_ylabel("Total PTM Mass Shift (Da)")
    ax.set_title("PTM Mass Shift by Charge State")
    save_plot(fig, os.path.join(output_dir, "boxplot_ptm_by_charge.png"))
    
    print(f"All visualizations have been saved in the '{output_dir}' folder.")

if __name__ == "__main__":
    main()
