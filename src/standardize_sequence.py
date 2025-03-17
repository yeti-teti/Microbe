import re
import logging
from typing import Dict, Optional

logger = logging.getLogger("ptm")

def standardize_sequence(sequence: str, residue_dict: Dict[str, float]) -> str:
    """
    Standardize a peptide sequence's modifications to match the exact format in config.yaml.
    
    This function handles small precision differences in modification masses
    (e.g., +57.021 vs +57.02146) by finding the closest matching entry in the residue dictionary.
    
    Args:
        sequence (str): The peptide sequence with modifications
        residue_dict (dict): Dictionary of residues from config.yaml
        
    Returns:
        str: Standardized sequence with modifications in the correct format
    """
    if not residue_dict:
        logger.warning("No residue dictionary provided for standardization")
        return sequence
        
    # Pattern to match amino acids with modifications
    pattern = r'([A-Z])(\+\d+\.\d+)'
    
    # Find all modifications in the sequence
    modified_sequence = sequence
    
    for match in re.finditer(pattern, sequence):
        aa = match.group(1)
        mod = match.group(2)
        full_mod = aa + mod
        
        # Check if this exact modification is in the residue dictionary
        if full_mod in residue_dict:
            continue  # No need to change
        
        # Extract the modification mass
        mod_mass = float(mod[1:])  # Remove the + sign
        
        # Find all modifications for this amino acid in the residue dictionary
        matching_mods = []
        for key in residue_dict:
            if not isinstance(key, str):
                continue
                
            # Match keys of the form "A+42.01" where A is the amino acid
            if key.startswith(aa + "+"):
                key_mod = key[len(aa):]
                try:
                    key_mass = float(key_mod[1:])  # Remove the + sign
                    matching_mods.append((key, key_mass))
                except ValueError:
                    continue
        
        if not matching_mods:
            logger.warning(f"No matching modification found for {full_mod}")
            continue
            
        # Find the closest match based on mass
        closest_key, _ = min(matching_mods, key=lambda x: abs(x[1] - mod_mass))
        
        # Replace the modification if it's a close match (within 0.01 Da)
        mod_key_mass = float(closest_key[len(aa)+1:])
        if abs(mod_key_mass - mod_mass) <= 0.01:
            start, end = match.span()
            modified_sequence = modified_sequence[:start] + closest_key + modified_sequence[end:]
            logger.info(f"Standardized modification: {full_mod} -> {closest_key}")
    
    return modified_sequence