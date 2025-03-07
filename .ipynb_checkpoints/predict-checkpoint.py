import os
import torch
import numpy as np
import argparse
from model import CasanovoPTM, MGF_Dataset, AA_VOCAB_SIZE, NUM_OFFSET_BINS

# Mapping: PAD=0, SOS=1, EOS=2, then A=3, C=4, etc.
idx_to_aa = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>"}
aa_list = list("ACDEFGHIKLMNPQRSTVWY")
for i, aa in enumerate(aa_list, start=3):
    idx_to_aa[i] = aa

def decode_sequence(token_list):
    seq = ""
    for t in token_list:
        if t == 2:  # EOS
            break
        if t in [0, 1]:
            continue
        seq += idx_to_aa[t]
    return seq

# Greedy decoder for inference.
def predict_sequence(model, peak_seq, peak_mask, max_len=36, device=torch.device("cuda")):
    model.eval()
    sos_token = 1
    generated = [sos_token]
    with torch.no_grad():
        for _ in range(max_len - 1):
            tgt = torch.tensor(generated, dtype=torch.long, device=device).unsqueeze(0)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            aa_logits, mod_logits = model(peak_seq, peak_mask, tgt, tgt_mask)
            next_token = int(torch.argmax(aa_logits[:, -1, :]).item())
            generated.append(next_token)
            if next_token == 2:
                break
    return generated

def load_spectra_from_mgf(mgf_filename):
    spectra = []
    current_spec = {}
    peaks = []
    with open(mgf_filename, "r") as f:
        for line in f:
            line = line.strip()
            if line == "BEGIN IONS":
                current_spec = {}
                peaks = []
            elif line.startswith("TITLE="):
                current_spec["title"] = line.split("=", 1)[1]
            elif line.startswith("PEPMASS="):
                current_spec["pepmass"] = float(line.split("=")[1])
            elif line.startswith("CHARGE="):
                ch = line.split("=")[1].replace("+", "")
                current_spec["charge"] = int(ch)
            elif line.startswith("SEQ="):
                current_spec["sequence"] = line.split("=", 1)[1]
            elif line == "END IONS":
                current_spec["mz"] = []
                current_spec["intensity"] = []
                for peak in peaks:
                    parts = peak.split()
                    current_spec["mz"].append(float(parts[0]))
                    current_spec["intensity"].append(float(parts[1]))
                current_spec["mz"] = np.array(current_spec["mz"], dtype=np.float32)
                current_spec["intensity"] = np.array(current_spec["intensity"], dtype=np.float32)
                spectra.append(current_spec)
            else:
                if line:
                    peaks.append(line)
    return spectra

def predict_and_save(model, mgf_input, output_file, device=torch.device("cuda")):
    dataset = MGF_Dataset(mgf_input)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = []
    for peak_seq, peak_mask, _, _ in loader:
        peak_seq = peak_seq.to(device)
        peak_mask = peak_mask.to(device)
        token_ids = predict_sequence(model, peak_seq, peak_mask, max_len=36, device=device)
        pred_seq = decode_sequence(token_ids)
        predictions.append(pred_seq)
    with open(output_file, "w") as f:
        for i, seq in enumerate(predictions):
            f.write(f"Spec_{i}\t{seq}\n")
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_mgf", type=str, required=True, help="Input MGF file for prediction")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (pretrained or fine-tuned)")
    parser.add_argument("--output", type=str, required=True, help="Output TSV file for predictions")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CasanovoPTM(AA_VOCAB_SIZE, NUM_OFFSET_BINS, d_model=512, nhead=8, num_layers=6).to(device)
    if os.path.exists(args.ckpt):
        import numpy as np
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
        state = torch.load(args.ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state, strict=False)
        print("Loaded checkpoint from", args.ckpt)
    else:
        print("Checkpoint not found.")
        exit(1)
    
    predict_and_save(model, args.input_mgf, args.output, device=device)
