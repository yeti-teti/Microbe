import glob
import logging
import os
import tempfile
import uuid
import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from depthcharge.data import AnnotatedSpectrumIndex, SpectrumIndex
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint

from ..config import Config
from ..data import ms_io
from .dataloaders import DeNovoDataModule
from .model import Spec2PepWithPTM


import numpy.core.multiarray

torch.serialization.add_safe_globals([np.core.multiarray.scalar, np.dtypes.Float64DType, np.dtype])

logger = logging.getLogger("ptm")

class ModelRunner:
    def __init__(
        self,
        config: Config,
        model_filename: Optional[str] = None,
    ) -> None:
        """Initialize a ModelRunner"""
        self.config = config
        self.model_filename = model_filename

        # Initialized later:
        self.tmp_dir = None
        self.trainer = None
        self.model = None
        self.loaders = None
        self.writer = None

        # Configure checkpoints.
        if config.save_top_k is not None:
            self.callbacks = [
                ModelCheckpoint(
                    dirpath=config.model_save_folder_path,
                    monitor="valid_CELoss",
                    mode="min",
                    save_top_k=config.save_top_k,
                )
            ]
        else:
            self.callbacks = None
    
    def __enter__(self):
        """Enter the context manager"""
        self.tmp_dir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Cleanup on exit"""
        self.tmp_dir.cleanup()
        self.tmp_dir = None
        if self.writer is not None:
            self.writer.save()

    def train(
        self,
        train_peak_path: Iterable[str],
        valid_peak_path: Iterable[str],
    ) -> None:
        self.initialize_trainer(train=True)
        self.initialize_model(train=True)
    
        train_index = self._get_index(train_peak_path, True, "training")
        valid_index = self._get_index(valid_peak_path, True, "validation")
        self.initialize_data_module(train_index, valid_index)
        self.loaders.setup()
    
        # Pass ckpt_path if a checkpoint is provided
        ckpt_path = self.model_filename if self.model_filename else None
        self.trainer.fit(
            self.model,
            self.loaders.train_dataloader(),
            self.loaders.val_dataloader(),
            ckpt_path=ckpt_path  # Add this to resume from checkpoint
        )

    def evaluate(self, peak_path: Iterable[str], output_eval: str = None) -> None:
        self.initialize_trainer(train=False)
        self.initialize_model(train=False)
    
        test_index = self._get_index(peak_path, True, "evaluation")
        self.initialize_data_module(test_index=test_index)
        self.loaders.setup(stage="test", annotated=True)
    
        # Run validation and capture metrics
        metrics = self.trainer.validate(self.model, self.loaders.test_dataloader())
    
        # Handle metrics as a list of dictionaries
        if isinstance(metrics, list) and len(metrics) > 0:
            # Assuming single dataloader, take the first dictionary
            metrics_dict = metrics[0]
        else:
            metrics_dict = metrics  # Fallback if it's already a dict (unlikely here)
    
        # Write metrics to the output file if specified
        if output_eval:
            try:
                with open(output_eval, 'w') as f:
                    for key, value in metrics_dict.items():
                        f.write(f"{key}: {value}\n")
            except Exception as e:
                logger.error(f"Failed to write to {output_eval}: {e}")
                raise
    
    def predict(self, peak_path: Iterable[str], output: str) -> None:
        self.writer = ms_io.MztabWriter(Path(output).with_suffix(".mztab"))
        self.writer.set_metadata(
            self.config,
            model=str(self.model_filename),
            config_filename=self.config.file,
        )

        self.initialize_trainer(train=False)
        self.initialize_model(train=False)
        self.model.out_writer = self.writer

        test_index = self._get_index(peak_path, False, "")
        self.writer.set_ms_run(test_index.ms_files)
        self.initialize_data_module(test_index=test_index)
        self.loaders.setup(stage="test", annotated=False)
        self.trainer.predict(self.model, self.loaders.test_dataloader())

    def initialize_trainer(self, train: bool) -> None:
        trainer_cfg = dict(
            accelerator=self.config.accelerator,
            devices=1,
            enable_checkpointing=False,
        )

        if train:
            if self.config.devices is None:
                devices = "auto"
            else:
                devices = self.config.devices

            additional_cfg = dict(
                devices=devices,
                callbacks=self.callbacks,
                enable_checkpointing=self.config.save_top_k is not None,
                max_epochs=self.config.max_epochs,
                num_sanity_val_steps=self.config.num_sanity_val_steps,
                strategy=self._get_strategy(),
                val_check_interval=self.config.val_check_interval,
                check_val_every_n_epoch=None,
            )
            trainer_cfg.update(additional_cfg)

        self.trainer = pl.Trainer(**trainer_cfg)
    
    def initialize_model(self, train: bool) -> None:

        model_params = dict(
            dim_model=self.config.dim_model,
            n_head=self.config.n_head,
            dim_feedforward=self.config.dim_feedforward,
            n_layers=self.config.n_layers,
            dropout=self.config.dropout,
            dim_intensity=self.config.dim_intensity,
            max_length=self.config.max_length,
            residues=self.config.residues,
            max_charge=self.config.max_charge,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            min_peptide_len=self.config.min_peptide_len,
            n_beams=self.config.n_beams,
            top_match=self.config.top_match,
            n_log=self.config.n_log,
            tb_summarywriter=self.config.tb_summarywriter,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            out_writer=self.writer,
            calculate_precision=self.config.calculate_precision,
        )

        # Reconfigurable non-architecture related parameters for a loaded model.
        loaded_model_params = dict(
            max_length=self.config.max_length,
            precursor_mass_tol=self.config.precursor_mass_tol,
            isotope_error_range=self.config.isotope_error_range,
            n_beams=self.config.n_beams,
            min_peptide_len=self.config.min_peptide_len,
            top_match=self.config.top_match,
            n_log=self.config.n_log,
            tb_summarywriter=self.config.tb_summarywriter,
            train_label_smoothing=self.config.train_label_smoothing,
            warmup_iters=self.config.warmup_iters,
            cosine_schedule_period_iters=self.config.cosine_schedule_period_iters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            out_writer=self.writer,
            calculate_precision=self.config.calculate_precision,
        )

        if self.model_filename is None:
            # Train a model from scratch if no model file is provided.
            if train:
                self.model = Spec2PepWithPTM(**model_params)
                return
            # Else we're not training, so a model file must be provided.
            else:
                logger.error("A model file must be provided")
                raise ValueError("A model file must be provided")
        # Else a model file is provided (to continue training or for inference).

        if not Path(self.model_filename).exists():
            logger.error(
                "Could not find the model weights at file %s",
                self.model_filename,
            )
            raise FileNotFoundError("Could not find the model weights file")

        # First try loading model details from the weights file, otherwise use the provided configuration.
        if self.model_filename is not None:
            if not Path(self.model_filename).exists():
                logger.error("Could not find the model weights at file %s", self.model_filename)
                raise FileNotFoundError("Could not find the model weights file")
    
            device = torch.empty(1).device  # Use the default device
            try:
                self.model = Spec2PepWithPTM.load_from_checkpoint(
                    self.model_filename, map_location=device, **loaded_model_params
                )
            except RuntimeError:
                try:
                    self.model = Spec2PepWithPTM.load_from_checkpoint(
                        self.model_filename,
                        map_location=device,
                        **model_params,
                    )
                except RuntimeError:
                    raise RuntimeError(
                        "Weights file incompatible"
                    )
    
    def initialize_data_module(
        self,
        train_index: Optional[AnnotatedSpectrumIndex] = None,
        valid_index: Optional[AnnotatedSpectrumIndex] = None,
        test_index: Optional[
            Union[AnnotatedSpectrumIndex, SpectrumIndex]
        ] = None,
    ) -> None:
        
        try:
            n_devices = self.trainer.num_devices
            train_bs = self.config.train_batch_size // n_devices
            eval_bs = self.config.predict_batch_size // n_devices
        except AttributeError:
            raise RuntimeError("Please use `initialize_trainer()` first.")

        self.loaders = DeNovoDataModule(
            train_index=train_index,
            valid_index=valid_index,
            test_index=test_index,
            min_mz=self.config.min_mz,
            max_mz=self.config.max_mz,
            min_intensity=self.config.min_intensity,
            remove_precursor_tol=self.config.remove_precursor_tol,
            n_workers=self.config.n_workers,
            train_batch_size=train_bs,
            eval_batch_size=eval_bs,
            residue_dict=self.config.residues,  # Pass the residue dictionary
        )
    
    def _get_index(
        self,
        peak_path: Iterable[str],
        annotated: bool,
        msg: str = "",
    ) -> Union[SpectrumIndex, AnnotatedSpectrumIndex]:
        
        ext = (".mgf", ".h5", ".hdf5")
        if not annotated:
            ext += (".mzml", ".mzxml")

        msg = msg.strip()
        filenames = _get_peak_filenames(peak_path, ext)
        if not filenames:
            not_found_err = f"Cound not find {msg} peak files"
            logger.error(not_found_err + " from %s", peak_path)
            raise FileNotFoundError(not_found_err)

        is_index = any([Path(f).suffix in (".h5", ".hdf5") for f in filenames])
        if is_index:
            if len(filenames) > 1:
                h5_err = f"Multiple {msg} HDF5 spectrum indexes specified"
                logger.error(h5_err)
                raise ValueError(h5_err)

            index_fname, filenames = filenames[0], None
        else:
            index_fname = Path(self.tmp_dir.name) / f"{uuid.uuid4().hex}.hdf5"

        Index = AnnotatedSpectrumIndex if annotated else SpectrumIndex
        valid_charge = np.arange(1, self.config.max_charge + 1)
        return Index(index_fname, filenames, valid_charge=valid_charge)
    
    def _get_strategy(self) -> Union[str, DDPStrategy]:
        if self.config.accelerator in ("cpu", "mps"):
            return "auto"
        elif self.config.devices == 1:
            return "auto"
        elif torch.cuda.device_count() > 1:
            return DDPStrategy(find_unused_parameters=False, static_graph=True)
        else:
            return "auto"
    
def _get_peak_filenames(
    paths: Iterable[str], supported_ext: Iterable[str]
) -> List[str]:
    found_files = set()
    for path in paths:
        path = os.path.expanduser(path)
        path = os.path.expandvars(path)
        for fname in glob.glob(path, recursive=True):
            if Path(fname).suffix.lower() in supported_ext:
                found_files.add(fname)

    return sorted(list(found_files))