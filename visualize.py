import matplotlib.pyplot as plt
import numpy as np
import re

def compute_fragment_masses(peptide):
    """Compute approximate b- and y-ion masses for a peptide.
       If modifications are present (e.g., M+15.995), include the mod mass.
    """
    aa_masses = {
        "A": 71.03711, "C": 103.00919, "D": 115.02694, "E": 129.04259,
        "F": 147.06841, "G": 57.02146, "H": 137.05891, "I": 113.08406,
        "K": 128.09496, "L": 113.08406, "M": 131.04049, "N": 114.04293,
        "P": 97.05276, "Q": 128.05858, "R": 156.10111, "S": 87.03203,
        "T": 101.04768, "V": 99.06841, "W": 186.07931, "Y": 163.06333
    }
    proton = 1.007276
    b_masses = []
    current = 0.0
    residues = re.findall(r"[A-Z](?:\+[\d\.]+)?", peptide)
    for res in residues:
        match = re.match(r"([A-Z])(\+[\d\.]+)?", res)
        if match:
            aa = match.group(1)
            mod = float(match.group(2)) if match.group(2) else 0.0
            current += aa_masses.get(aa, 0.0) + mod
            b_masses.append(current)
    total_mass = current + 18.01056
    y_masses = [total_mass - b for b in b_masses]
    return {"b": b_masses, "y": y_masses}

def plot_spectrum_with_annotation(spec, predicted_seq):
    """
    Plot an MS/MS spectrum with annotated b- and y-ion peaks based on the predicted peptide.
    spec: dict with keys 'mz', 'intensity', 'title'
    predicted_seq: predicted peptide string (e.g., 'PEPM+15.995IDEK')
    """
    mz = spec["mz"]
    intensity = spec["intensity"]
    title = spec.get("title", "Spectrum")
    fragments = compute_fragment_masses(predicted_seq)
    b_masses = fragments["b"]
    y_masses = fragments["y"]
    
    plt.figure(figsize=(10,6))
    plt.stem(mz, intensity, basefmt=" ", use_line_collection=True, label="Observed")
    for i, m in enumerate(b_masses):
        idx = np.argmin(np.abs(mz - m))
        if abs(mz[idx] - m) < 0.5:
            plt.text(mz[idx], intensity[idx]*1.1, f"b{i+1}", rotation=90, color="blue")
    for i, m in enumerate(y_masses):
        idx = np.argmin(np.abs(mz - m))
        if abs(mz[idx] - m) < 0.5:
            plt.text(mz[idx], intensity[idx]*1.1, f"y{i+1}", rotation=90, color="red")
    plt.title(f"{title}\nPredicted: {predicted_seq}")
    plt.xlabel("m/z")
    plt.ylabel("Normalized Intensity")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example: visualize a sample spectrum.
    sample_spec = {
        "title": "Spec_0",
        "mz": np.array([100.0, 150.0, 200.0, 250.0, 300.0]),
        "intensity": np.array([0.1, 0.4, 0.3, 0.15, 0.05]),
        "pepmass": 600.0
    }
    predicted_seq = "PEPM+15.995IDEK"
    plot_spectrum_with_annotation(sample_spec, predicted_seq)
