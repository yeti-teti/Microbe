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
                f"Casanovo identification file "
                f"{os.path.splitext(os.path.basename(self.filename))[0]}",
            ),
            (
                "psm_search_engine_score[1]",
                "[MS, MS:1001143, search engine specific score for PSMs, ]",
            ),
        ]
        self._run_map = {}
        self.psms = []
        self.residues = None  # Will be set in set_metadata
        self.standard_masses = None  # Will be set in set_metadata

    def set_metadata(self, config: Config, **kwargs) -> None:
        """
        Specify metadata information to write to the mzTab header.

        Parameters
        ----------
        config : Config
            The active configuration options.
        kwargs
            Additional configuration options (i.e. from command-line arguments).
        """
        known_mods = {
            "+57.021": "[UNIMOD, UNIMOD:4, Carbamidomethyl, ]",
            "+15.995": "[UNIMOD, UNIMOD:35, Oxidation, ]",
            "+0.984": "[UNIMOD, UNIMOD:7, Deamidated, ]",
            "+42.011": "[UNIMOD, UNIMOD:1, Acetyl, ]",
            "+43.006": "[UNIMOD, UNIMOD:5, Carbamyl, ]",
            "-17.027": "[UNIMOD, UNIMOD:385, Ammonia-loss, ]",
        }
        residues = collections.defaultdict(set)
        for aa, mass in config["residues"].items():
            aa_mod = re.match(r"([A-Z]?)([+-]?(?:[0-9]*[.])?[0-9]+)", aa)
            if aa_mod is None:
                residues[aa].add(None)
            else:
                residues[aa_mod[1]].add(aa_mod[2])
        fixed_mods, variable_mods = [], []
        for aa, mods in residues.items():
            if len(mods) > 1:
                for mod in mods:
                    if mod is not None:
                        variable_mods.append((aa, mod))
            elif None not in mods:
                fixed_mods.append((aa, mods.pop()))

        if len(fixed_mods) == 0:
            self.metadata.append(
                (
                    "fixed_mod[1]",
                    "[MS, MS:1002453, No fixed modifications searched, ]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(fixed_mods, 1):
                self.metadata.append(
                    (
                        f"fixed_mod[{i}]",
                        known_mods.get(mod, f"[CHEMMOD, CHEMMOD:{mod}, , ]"),
                    )
                )
                self.metadata.append(
                    (f"fixed_mod[{i}]-site", aa if aa else "N-term")
                )
        if len(variable_mods) == 0:
            self.metadata.append(
                (
                    "variable_mod[1]",
                    "[MS, MS:1002454, No variable modifications searched,]",
                )
            )
        else:
            for i, (aa, mod) in enumerate(variable_mods, 1):
                self.metadata.append(
                    (
                        f"variable_mod[{i}]",
                        known_mods.get(mod, f"[CHEMMOD, CHEMMOD:{mod}, , ]"),
                    )
                )
                self.metadata.append(
                    (f"variable_mod[{i}]-site", aa if aa else "N-term")
                )
        for i, (key, value) in enumerate(kwargs.items(), 1):
            self.metadata.append(
                (f"software[1]-setting[{i}]", f"{key} = {value}")
            )
        for i, (key, value) in enumerate(config.items(), len(kwargs) + 1):
            if key not in ("residues",):
                self.metadata.append(
                    (f"software[1]-setting[{i}]", f"{key} = {value}")
                )

        # Store residues and standard masses for PTM calculations and parsing
        self.residues = config["residues"]
        self.standard_masses = {aa: mass for aa, mass in self.residues.items() if re.match(r"^[A-Z]$", aa)}

    def set_ms_run(self, peak_filenames: List[str]) -> None:
        """
        Add input peak files to the mzTab metadata section.

        Parameters
        ----------
        peak_filenames : List[str]
            The input peak file name(s).
        """
        for i, filename in enumerate(natsort.natsorted(peak_filenames), 1):
            filename = os.path.abspath(filename)
            self.metadata.append(
                (f"ms_run[{i}]-location", Path(filename).as_uri()),
            )
            self._run_map[filename] = i

    def parse_sequence(self, sequence):
        """
        Parse a peptide sequence into a list of residues (standard or modified).
        
        Args:
            sequence (str): Predicted sequence (e.g., "AMC+57.02146K+42.010565").
        
        Returns:
            list: List of parsed residues (e.g., ["A", "M", "C+57.02146", "K+42.010565"]).
        """
        residue_list = sorted(self.residues.keys(), key=lambda x: -len(x))
        parsed = []
        while sequence:
            for res in residue_list:
                if sequence.startswith(res):
                    parsed.append(res)
                    sequence = sequence[len(res):]
                    break
            else:
                raise ValueError(f"Cannot parse sequence: {sequence}")
        return parsed

    def get_base_residue(self, res):
        """
        Extract the base amino acid from a residue (modified or unmodified).
        
        Args:
            res (str): Residue name (e.g., "C+57.02146" or "A").
        
        Returns:
            str: Base amino acid (e.g., "C" or "A").
        """
        if "+" in res or "-" in res:
            return re.match(r'^[A-Z]', res).group(0)
        return res

    def calculate_base_mass(self, parsed_sequence):
        """
        Calculate the base mass of a peptide by summing the masses of its base residues.
        
        Args:
            parsed_sequence (list): List of residues (e.g., ["A", "M", "C+57.02146"]).
        
        Returns:
            float: Total base mass in Daltons.
        """
        base_mass = 0.0
        for res in parsed_sequence:
            base_aa = self.get_base_residue(res)
            if base_aa in self.standard_masses:
                base_mass += self.standard_masses[base_aa]
            else:
                raise ValueError(f"Unknown base amino acid: {base_aa}")
        return base_mass

    def calculate_total_ptm_mass_shift(self, psm):
        """
        Calculate the total PTM mass shift for a PSM.
        
        Args:
            psm (tuple): PSM data with indices:
                - 0: sequence (str)
                - 3: charge (int)
                - 5: calc_mass_to_charge (float)
        
        Returns:
            float: Total PTM mass shift in Daltons.
        """
        sequence = psm[0]
        charge = int(psm[3])
        calc_mass_to_charge = float(psm[5])
        proton_mass = 1.007825  # Mass of a proton in Daltons
        parsed_sequence = self.parse_sequence(sequence)
        base_mass = self.calculate_base_mass(parsed_sequence)
        calculated_neutral_mass = charge * calc_mass_to_charge - charge * proton_mass
        total_ptm_mass_shift = calculated_neutral_mass - base_mass
        return total_ptm_mass_shift

    def get_mod_id(self, mass_shift_str):
        """
        Map a mass shift to a modification identifier.
        
        Args:
            mass_shift_str (str): Mass shift, e.g., "57.02146".
        
        Returns:
            str: Modification ID, e.g., "[UNIMOD, UNIMOD:4, Carbamidomethyl, ]".
        """
        try:
            mass_shift = float(mass_shift_str)
        except ValueError:
            return f"[CHEMMOD, CHEMMOD:+{mass_shift_str}, , ]"
        
        # Define known modifications with exact mass shifts
        if abs(mass_shift - 15.994915) < 0.001:
            return "[UNIMOD, UNIMOD:35, Oxidation, ]"
        elif abs(mass_shift - 57.02146) < 0.001:
            return "[UNIMOD, UNIMOD:4, Carbamidomethyl, ]"
        elif abs(mass_shift - 42.010565) < 0.001:
            return "[UNIMOD, UNIMOD:1, Acetyl, ]"
        # Add more modifications as needed based on your config
        else:
            return f"[CHEMMOD, CHEMMOD:+{mass_shift}, , ]"

    def get_modifications(self, parsed_sequence):
        """
        Generate the modifications string for mzTab.
        
        Args:
            parsed_sequence (list): Parsed residues, e.g., ["A", "C+57.02146", "K"].
        
        Returns:
            str: Modifications string, e.g., "2-[UNIMOD, UNIMOD:4, Carbamidomethyl, ]".
        """
        modifications = []
        for pos, res in enumerate(parsed_sequence, 1):  # 1-based indexing for mzTab
            if "+" in res:
                parts = res.split("+")
                base_aa = parts[0]  # e.g., "C"
                for mod in parts[1:]:  # e.g., ["57.02146"]
                    mod_id = self.get_mod_id(mod)
                    modifications.append(f"{pos}-{mod_id}")
        return ",".join(modifications) if modifications else "null"

    def save(self) -> None:
        """
        Export the spectrum identifications to the mzTab file.
        """
        with open(self.filename, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator=os.linesep)
            # Write metadata
            for row in self.metadata:
                writer.writerow(["MTD", *row])
            # Write PSM header
            writer.writerow(
                [
                    "PSH",
                    "sequence",
                    "PSM_ID",
                    "accession",
                    "unique",
                    "database",
                    "database_version",
                    "search_engine",
                    "search_engine_score[1]",
                    "modifications",
                    "retention_time",
                    "charge",
                    "exp_mass_to_charge",
                    "calc_mass_to_charge",
                    "spectra_ref",
                    "pre",
                    "post",
                    "start",
                    "end",
                    "opt_ms_run[1]_aa_scores",
                    "total_ptm_mass_shift",
                ]
            )
            # Write PSM data
            for i, psm in enumerate(
                natsort.natsorted(self.psms, key=operator.itemgetter(1)), 1
            ):
                filename, idx = os.path.abspath(psm[1][0]), psm[1][1]
                sequence = psm[0]  # Peptide sequence
                parsed_sequence = self.parse_sequence(sequence)
                modifications = self.get_modifications(parsed_sequence)
                writer.writerow(
                    [
                        "PSM",
                        sequence,              # Peptide sequence
                        i,                     # PSM_ID
                        "null",                # accession
                        "null",                # unique
                        "null",                # database
                        "null",                # database_version
                        psm[2],                # search_engine_score[1]
                        modifications,         # PTMs extracted from sequence
                        "null",                # retention_time
                        psm[3],                # charge
                        psm[4],                # exp_mass_to_charge
                        psm[5],                # calc_mass_to_charge
                        f"ms_run[{self._run_map[filename]}]:{idx}",
                        "null",                # pre
                        "null",                # post
                        "null",                # start
                        "null",                # end
                        psm[6],                # opt_ms_run[1]_aa_scores
                        str(self.calculate_total_ptm_mass_shift(psm)),  # total_ptm_mass_shift
                    ]
                )