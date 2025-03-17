import logging
import shutil
import warnings
import os
import yaml
import torch
import numpy as np
import random
from pathlib import Path
from typing import Optional, Dict, Callable, Tuple, Union
from datetime import datetime

logger = logging.getLogger("ptm")

_config_deprecated = dict(
    every_n_train_steps="val_check_interval",
    max_iters="cosine_schedule_period_iters",
)

class Config:
    _default_config = Path(__file__).parent / "config.yaml"
    _config_types = dict(
        random_seed=int,
        n_peaks=int,
        min_mz=float,
        max_mz=float,
        min_intensity=float,
        remove_precursor_tol=float,
        max_charge=int,
        precursor_mass_tol=float,
        isotope_error_range=lambda min_max: (int(min_max[0]), int(min_max[1])),
        min_peptide_len=int,
        dim_model=int,
        n_head=int,
        dim_feedforward=int,
        n_layers=int,
        dropout=float,
        dim_intensity=int,
        max_length=int,
        residues=dict,
        n_log=int,
        tb_summarywriter=str,
        train_label_smoothing=float,
        warmup_iters=int,
        cosine_schedule_period_iters=int,
        learning_rate=float,
        weight_decay=float,
        train_batch_size=int,
        predict_batch_size=int,
        n_beams=int,
        top_match=int,
        max_epochs=int,
        num_sanity_val_steps=int,
        save_top_k=int,
        model_save_folder_path=str,
        val_check_interval=int,
        calculate_precision=bool,
        accelerator=str,
        devices=int,
    )

    def __init__(self, config_file: Optional[str] = None, **kwargs):
        """
        Initialize the Config object.
        
        Args:
            config_file (str, optional): Path to the config file. Defaults to None.
            **kwargs: Additional arguments to override the config file.
        """
        # Initialize _params dictionary where we'll store all parameters
        self._params = {}
        
        # Load default parameters
        self._user_config = {}
        self._load_default_config()
        
        # Load from config file if provided
        if config_file is not None:
            self.file = config_file
            with open(config_file, "r") as f:
                self._user_config = yaml.safe_load(f)
        else:
            self.file = None
            
        # Apply the configuration
        self._apply_config()
        
        # Override with kwargs
        for key, value in kwargs.items():
            self._params[key] = value
            
        # Initialize additional parameters that were in the original constructor
        self.n_workers = os.cpu_count()
    
    def _load_default_config(self):
        """Load the default configuration from the YAML file."""
        try:
            with open(self._default_config, "r") as f:
                self._params = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning("Default config file not found. Using empty configuration.")
            self._params = {}
    
    def _apply_config(self):
        """Apply configuration by validating parameters."""
        for param, param_type in self._config_types.items():
            self.validate_param(param, param_type)
    
    def __getitem__(self, param: str) -> Union[int, bool, str, Tuple, Dict]:
        """Retrieve a parameter"""
        return self._params.get(param)

    def __getattr__(self, param: str) -> Union[int, bool, str, Tuple, Dict]:
        """Retrieve a parameter"""
        if param in self._params:
            return self._params.get(param)
        raise AttributeError(f"'Config' object has no attribute '{param}'")

    def validate_param(self, param: str, param_type: Callable):
        try:
            param_val = self._user_config.get(param, self._params.get(param))
            if param == "residues":
                residues = {
                    str(aa): float(mass) for aa, mass in param_val.items()
                } if param_val is not None else {}
                self._params["residues"] = residues
            elif param_val is not None:
                self._params[param] = param_type(param_val)
        except (TypeError, ValueError) as err:
            logger.error(
                "Incorrect type for configuration value %s: %s", param, err
            )
            raise TypeError(
                f"Incorrect type for configuration value {param}: {err}"
            )

    def items(self) -> Tuple[str, ...]:
        """Return the parameters"""
        return self._params.items()

    @classmethod
    def copy_default(cls, output: str) -> None:
        """Copy the default YAML configuration.

        Parameters
        ----------
        output : str
            The output file.
        """
        shutil.copyfile(cls._default_config, output)