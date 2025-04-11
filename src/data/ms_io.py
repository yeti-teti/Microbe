#!/usr/bin/env python3
import collections
import csv
import operator
import os
import re
from pathlib import Path
from typing import List

import natsort

from src.config import Config

class MztabWriter:
    """
        Export spectrum identifications to an mzTab file.
    
        Parameters
        ----------
        filename : str
            The name of the mzTab file.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.metadata = [
            ("mzTab-version", "1.0.0"),
            ("mzTab-mode", "Summary"),
            ("mzTab-type", "Identification"),
            (
                "description",
                f"Data identification file {os.path.splitext(os.path.basename(self.filename))[0]}",
            ),
            ("psm_search_engine_score[1]", "[MS, MS:1001143, search engine specific score for PSMs, ]"),
        ]
        self._run_map = {}
        self.psms = []
        self.residues = None           # This will be set in set_metadata from config
        self.standard_masses = None    # Map of base residue letter to its mass
        self.modifications = {}        # Dynamic mapping: modification mass (as string) -> modification annotation

    def set_metadata(self, config: Config, **kwargs) -> None:
        """
            Specify metadata information to write to the mzTab header.
        """
        for i, (key, value) in enumerate(kwargs.items(), 1):
            self.metadata.append((f"software[1]-setting[{i}]", f"{key} = {value}"))
        for i, (key, value) in enumerate(config.items(), len(kwargs) + 1):
            if key != "residues":
                self.metadata.append((f"software[1]-setting[{i}]", f"{key} = {value}"))

        # Store the entire residues dictionary from config
        self.residues = config["residues"]

        # Build standard_masses from those keys that are a single letter (the unmodified residue)
        self.standard_masses = {aa: mass for aa, mass in self.residues.items() if re.fullmatch(r"[A-Z]", aa)}

        # Build a dynamic mapping for modifications.
        # For keys that represent a single modification (e.g. "C+57.02146" or "M-17.027"),extract the modification mass difference and use it to build an annotation.
        # If a key contains more than one modification (e.g., "A+15.994915+57.02146"),ignore it as the individual modifications will be parsed later.
        self.modifications = {}
        mod_pattern = re.compile(r"^[A-Z]([+-]\d+(?:\.\d+)?)$")
        for key, mass in self.residues.items():
            match = mod_pattern.fullmatch(key)
            if match:
                # key format is e.g., "C+57.02146"
                mod_str = match.group(1)  # e.g., "+57.02146"
                try:
                    mod_val = float(mod_str)  # + sign is handled by float conversion
                except ValueError:
                    continue
                # standard base mass for the residue to compute the difference.
                base = key[0]
                if base in self.standard_masses:
                    diff = mod_val  # Since the key is "base+mod", the modification is simply mod_val.
                    diff_key = f"{round(diff, 5):.5f}"
                    self.modifications[diff_key] = f"[CHEMMOD, CHEMMOD:+{diff_key}, , ]"
        if self.modifications:
            for i, (mod_mass, mod_annotation) in enumerate(sorted(self.modifications.items()), 1):
                self.metadata.append((f"fixed_mod[{i}]", mod_annotation))
                self.metadata.append((f"fixed_mod[{i}]-site", "N-term"))  
        else:
            self.metadata.append(("fixed_mod[1]", "[MS, MS:1002453, No fixed modifications searched, ]"))

    def set_ms_run(self, peak_filenames: List[str]) -> None:
        """
        Add input peak files to the mzTab metadata section.
        """
        for i, filename in enumerate(natsort.natsorted(peak_filenames), 1):
            filename = os.path.abspath(filename)
            self.metadata.append((f"ms_run[{i}]-location", Path(filename).as_uri()))
            self._run_map[filename] = i

    def parse_sequence(self, sequence):
        """
        Parse a peptide sequence into tokens.
        Uses a regex that matches a single uppercase letter followed by zero or more modifications.
        E.g., "AFYAAVAADC+57.02146FR" is split into:
              ["A", "F", "Y", "A", "A", "V", "A", "A", "D", "C+57.02146", "F", "R"]
        """
        pattern = r"[A-Z](?:[+-]\d+(?:\.\d+)?)*"
        tokens = re.findall(pattern, sequence)
        if not tokens:
            raise ValueError(f"Cannot parse sequence: {sequence}")
        return tokens

    def get_base_residue(self, token):
        """
        Return the base amino acid from a token.
        """
        match = re.match(r"^([A-Z])", token)
        if match:
            return match.group(1)
        raise ValueError(f"Invalid residue token: {token}")

    def calculate_base_mass(self, parsed_sequence):
        """
        Calculate the base mass of a peptide by summing the masses of its base residues.
        """
        base_mass = 0.0
        for token in parsed_sequence:
            base_aa = self.get_base_residue(token)
            if base_aa in self.standard_masses:
                base_mass += self.standard_masses[base_aa]
            else:
                raise ValueError(f"Unknown base amino acid: {base_aa}")
        return base_mass

    def calculate_total_ptm_mass_shift(self, psm):
        """
        Calculate the total PTM mass shift for a PSM.
        """
        sequence = psm[0]
        charge = int(psm[3])
        calc_mass_to_charge = float(psm[5])
        proton_mass = 1.007825  # Da
        parsed_sequence = self.parse_sequence(sequence)
        base_mass = self.calculate_base_mass(parsed_sequence)
        calculated_neutral_mass = charge * calc_mass_to_charge - charge * proton_mass
        total_ptm_mass_shift = calculated_neutral_mass - base_mass
        return total_ptm_mass_shift

    def get_mod_id(self, mass_shift_str):
        """
        Dynamically map a mass shift string to its modification annotation based on the config.
        """
        try:
            mass_shift = float(mass_shift_str)
        except ValueError:
            return f"[CHEMMOD, CHEMMOD:+{mass_shift_str}, , ]"
        diff_key = f"{round(mass_shift,5):.5f}"
        if self.modifications and diff_key in self.modifications:
            return self.modifications[diff_key]
        else:
            return f"[CHEMMOD, CHEMMOD:+{diff_key}, , ]"

    def get_modifications(self, parsed_sequence):
        """
        Generate the modifications string for mzTab.
        For each token (e.g., "C+57.02146+15.994915"), extract all modification values.
        Returns a comma-separated list in the format:
           "<position>-[modification ID]"
        """
        modifications = []
        # Use regex to capture all signed numbers (e.g., +57.02146 or -17.027)
        mod_pattern = re.compile(r"([+-]\d+(?:\.\d+)?)")
        for pos, token in enumerate(parsed_sequence, 1):
            mods = mod_pattern.findall(token)
            for mod in mods:
                # Remove the leading '+' (if desired)
                mod_clean = mod[1:] if mod.startswith('+') else mod
                mod_id = self.get_mod_id(mod_clean)
                modifications.append(f"{pos}-{mod_id}")
        return ",".join(modifications) if modifications else "null"

    def save(self) -> None:
        """
        Export the spectrum identifications to the mzTab file.
        """
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter="*", lineterminator=os.linesep)
            # Write metadata lines
            for row in self.metadata:
                writer.writerow(["MTD", *row])
            # Write PSH header line
            writer.writerow([
                "PSH", "sequence", "PSM_ID", "accession", "unique",
                "database", "database_version", "search_engine",
                "search_engine_score[1]", "modifications", "retention_time",
                "charge", "exp_mass_to_charge", "calc_mass_to_charge",
                "spectra_ref", "pre", "post", "start", "end",
                "opt_ms_run[1]_aa_scores", "total_ptm_mass_shift",
            ])
            # Write each PSM
            for i, psm in enumerate(natsort.natsorted(self.psms, key=operator.itemgetter(1)), 1):
                filename, idx = os.path.abspath(psm[1][0]), psm[1][1]
                sequence = psm[0]
                parsed_sequence = self.parse_sequence(sequence)
                modifications = self.get_modifications(parsed_sequence)
                writer.writerow([
                    "PSM",
                    sequence,                           # Peptide sequence
                    i,                                  # PSM_ID
                    "null",                             # accession
                    "null",                             # unique
                    "null",                             # database
                    "null",                             # database_version
                    "null",                             # search engine
                    psm[2],                             # search_engine_score[1]
                    modifications,                      # modifications string
                    "null",                             # retention_time
                    psm[3],                             # charge
                    psm[4],                             # exp_mass_to_charge
                    psm[5],                             # calc_mass_to_charge
                    f"ms_run[{self._run_map[filename]}]:{idx}",
                    "null",                             # pre
                    "null",                             # post
                    "null",                             # start
                    "null",                             # end
                    psm[6],                             # opt_ms_run[1]_aa_scores
                    str(self.calculate_total_ptm_mass_shift(psm))  # total_ptm_mass_shift
                ])
