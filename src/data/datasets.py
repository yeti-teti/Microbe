from typing import Optional, Tuple

import depthcharge
import numpy as np
import spectrum_utils.spectrum as sus
import torch
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):
    def __init__(
        self,
        spectrum_index: depthcharge.data.SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        """Initialize a SpectrumDataset"""
        super().__init__()
        self.n_peaks = n_peaks
        self.min_mz = min_mz
        self.max_mz = max_mz
        self.min_intensity = min_intensity
        self.remove_precursor_tol = remove_precursor_tol
        self.rng = np.random.default_rng(random_state)
        self._index = spectrum_index

    def __len__(self) -> int:
        """The number of spectra."""
        return self.n_spectra
    
    def __getitem__(
        self, idx
    ) -> Tuple[torch.Tensor, float, int, Tuple[str, str]]:
        mz_array, int_array, precursor_mz, precursor_charge = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        return (
            spectrum,
            precursor_mz,
            precursor_charge,
            self.get_spectrum_id(idx),
        )
    
    def get_spectrum_id(self, idx: int) -> Tuple[str, str]:
        with self.index:
            return self.index.get_spectrum_id(idx)
        
    def _process_peaks(self, mz_array, int_array, precursor_mz, precursor_charge):
        spectrum = sus.MsmsSpectrum(
            "", precursor_mz, precursor_charge,
            mz_array.astype(np.float64), int_array.astype(np.float32)
        )
        try:
            spectrum.set_mz_range(self.min_mz, self.max_mz)
            if len(spectrum.mz) == 0:
                raise ValueError("No peaks in mz range")
            spectrum.remove_precursor_peak(self.remove_precursor_tol, "Da")
            if len(spectrum.mz) == 0:
                raise ValueError("No peaks after precursor removal")
            spectrum.filter_intensity(self.min_intensity, self.n_peaks)
            if len(spectrum.mz) == 0:
                raise ValueError("No peaks after intensity filter")
            spectrum.scale_intensity("root", 1)
            norm = np.linalg.norm(spectrum.intensity)
            if norm == 0:
                raise ValueError("Zero norm intensity")
            intensities = spectrum.intensity / norm
            return torch.tensor(np.array([spectrum.mz, intensities])).T.float()
        except ValueError as e:
            print(f"Warning: Spectrum processed as dummy - {str(e)}")
            return torch.tensor([[0, 1]]).float()
    
    @property
    def n_spectra(self) -> int:
        """The total number of spectra."""
        return self.index.n_spectra

    @property
    def index(self) -> depthcharge.data.SpectrumIndex:
        """The underlying SpectrumIndex."""
        return self._index

    @property
    def rng(self):
        """The NumPy random number generator."""
        return self._rng

    @rng.setter
    def rng(self, seed):
        """Set the NumPy random number generator."""
        self._rng = np.random.default_rng(seed)
    

class AnnotatedSpectrumDataset(SpectrumDataset):
    def __init__(
        self,
        annotated_spectrum_index: depthcharge.data.SpectrumIndex,
        n_peaks: int = 150,
        min_mz: float = 140.0,
        max_mz: float = 2500.0,
        min_intensity: float = 0.01,
        remove_precursor_tol: float = 2.0,
        random_state: Optional[int] = None,
    ):
        super().__init__(
            annotated_spectrum_index,
            n_peaks=n_peaks,
            min_mz=min_mz,
            max_mz=max_mz,
            min_intensity=min_intensity,
            remove_precursor_tol=remove_precursor_tol,
            random_state=random_state,
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, float, int, str]:
        (
            mz_array,
            int_array,
            precursor_mz,
            precursor_charge,
            peptide,
        ) = self.index[idx]
        spectrum = self._process_peaks(
            mz_array, int_array, precursor_mz, precursor_charge
        )
        return spectrum, precursor_mz, precursor_charge, peptide
    