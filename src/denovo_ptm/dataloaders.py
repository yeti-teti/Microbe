import functools
import os
import logging
from typing import List, Optional, Tuple, Dict, Union

import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.data import AnnotatedSpectrumIndex

from ..data.datasets import AnnotatedSpectrumDataset, SpectrumDataset

# Import the standardization function
try:
    from ..standardize_sequence import standardize_sequence
except ImportError:
    # Fallback if the import path doesn't work
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from standardize_sequence import standardize_sequence

logger = logging.getLogger("ptm")

class DeNovoDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_index: Optional[AnnotatedSpectrumIndex] = None,
        valid_index: Optional[AnnotatedSpectrumIndex] = None,
        test_index: Optional[AnnotatedSpectrumIndex] = None,
        train_batch_size: int = 128,
        eval_batch_size: int = 1028,
        n_peaks: Optional[int] = 150,
        min_mz: float = 50.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        n_workers: Optional[int] = None,
        random_state: Optional[int] = None,
        residue_dict: Optional[Dict[str, float]] = None,  # Add residue_dict parameter
    ):
        super().__init__()
        self.train_index = train_index
        self.valid_index = valid_index
        self.test_index = test_index
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.rng = np.random.default_rng(random_state)
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.residue_dict = residue_dict  # Store the residue dictionary
    
    def setup(self, stage: str = None, annotated: bool = True) -> None:
        if stage in (None, "fit", "validate"):
            make_dataset = functools.partial(
                AnnotatedSpectrumDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                remove_precursor_tol=self.remove_precursor_tol,
            )
            if self.train_index is not None:
                self.train_dataset = make_dataset(
                    self.train_index,
                    random_state=self.rng,
                )
            if self.valid_index is not None:
                self.valid_dataset = make_dataset(self.valid_index)
        if stage in (None, "test"):
            make_dataset = functools.partial(
                AnnotatedSpectrumDataset if annotated else SpectrumDataset,
                n_peaks=self.n_peaks,
                min_mz=self.min_mz,
                max_mz=self.max_mz,
                min_intensity=self.min_intensity,
                remove_precursor_tol=self.remove_precursor_tol,
            )
            if self.test_index is not None:
                self.test_dataset = make_dataset(self.test_index)
    
    def _make_loader(
        self,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        # Create a custom prepare_batch function with the residue_dict
        prepare_batch_with_residues = functools.partial(
            prepare_batch, 
            residue_dict=self.residue_dict
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=prepare_batch_with_residues,  # Use the customized function
            pin_memory=False,
            num_workers=self.n_workers,
            shuffle=shuffle,
        )
    
    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the training DataLoader."""
        return self._make_loader(
            self.train_dataset, self.train_batch_size, shuffle=True
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the validation DataLoader."""
        return self._make_loader(self.valid_dataset, self.eval_batch_size)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the test DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        """Get the predict DataLoader."""
        return self._make_loader(self.test_dataset, self.eval_batch_size)


def prepare_batch(
    batch: List[Tuple[torch.Tensor, float, int, str]],
    residue_dict=None
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    spectra, precursor_mzs, precursor_charges, spectrum_ids = list(zip(*batch))
    
    # Standardize peptide sequences if residue_dict is provided
    if residue_dict is not None and all(isinstance(seq, str) for seq in spectrum_ids):
        try:
            from src.standardize_sequence import standardize_sequence
            spectrum_ids = [standardize_sequence(seq, residue_dict) for seq in spectrum_ids]
        except Exception as e:
            print(f"Error standardizing sequences: {e}")
    
    spectra = torch.nn.utils.rnn.pad_sequence(spectra, batch_first=True)
    precursor_mzs = torch.tensor(precursor_mzs)
    precursor_charges = torch.tensor(precursor_charges)
    precursor_masses = (precursor_mzs - 1.007276) * precursor_charges
    precursors = torch.vstack(
        [precursor_masses, precursor_charges, precursor_mzs]
    ).T.float()
    
    return spectra, precursors, np.asarray(spectrum_ids)