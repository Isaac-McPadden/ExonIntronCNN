# experiment_framework.py
"""
High-level experiment orchestration for the Intron-Exon project.

Each CSV row represents a *trial* that is trained for a fixed number of
**epoch-units** (``config.EPOCH_UNITS_PER_TRIAL``).  One epoch-unit corresponds
exactly to ``config.STEPS_PER_EPOCH_UNIT`` gradient-descent steps on the
training dataset.

Responsibilities
----------------
*     Experiment folder setup and per-trial sub-folders
*     Incremental/resumable training with checkpoint discovery
*     Loading TFRecord datasets based on naming conventions
*     Selecting the appropriate model architecture, loss, metrics and callbacks
*     Aggregating Keras ``History`` objects across epoch-units and persisting
      them crash-safely as JSON
*     Re-plotting a train/val loss curve and saving a final model artefact at
      the end of every trial

Revision history
----------------
🔄 **2025-04-29 rewrite** —⁠ consolidated fixes discussed with ChatGPT:
    • Added missing imports, Path-safe ops, and metrics instantiation
    • Numeric checkpoint ordering
    • One-file running history with crash-safety
    • Correct ``epochs``/``initial_epoch`` per unit
    • Avoid mutating global callback list; TB/Updater added per unit
    • LR-scheduler state kept *per-trial*
    • ``_dataset_path_from_row`` uses ``self.base_data_dir``
    • Garbage-collect & clear Keras session between trials
"""
from __future__ import annotations

import json
import re
import tempfile
import shutil
import gc
import copy
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import pandas as pd
import tensorflow as tf
from keras import callbacks, optimizers, models, backend, utils

# ── In-house modules ───────────────────────────────────────────────────────────
from IEModules import (
    Data_Functions,
    Custom_Models,
    Custom_Callbacks,
    Custom_Losses,
    Helper_Functions,
    Custom_Metrics,
    Custom_Optimizers,
)
from IEModules.config import (
    MODEL_DIR,
    LOG_DIR,
    EPOCH_UNITS_PER_TRIAL,
    PHYSICAL_BATCH_SIZE,
    STEPS_PER_EPOCH_UNIT,
    experiment_data_folder,
    LR_STATE_SAVE_SUBDIR,
    LR_STATE_SAVE_FILENAME,
)
from IEModules import config as cfg  # Generic access to loss hyper-params

# ── Dataset naming helpers ─────────────────────────────────────────────────────

def build_dataset_filepath(
    split: str,
    units: int,
    style_flag: str,
    smoothing_flag: str,
    base_data_dir: Path = experiment_data_folder,
) -> Path:
    """Resolve the unique TFRecord file that matches a naming convention.

    Parameters
    ----------
    split : str
        One of ``{"train", "val", "test"}``.
    units : int
        Size flag taken from the CSV (number of epoch-units included when the
        TFRecord was generated).
    style_flag : str
        Either ``"M"`` (mixed) or ``"I"`` (incremental).  Only the first
        character is used.
    smoothing_flag : str
        Either ``"C"`` (custom smoothed labels) or ``"B"`` (binary labels).
    base_data_dir : pathlib.Path, optional
        Directory that contains the dataset files.  Defaults to the folder
        chosen in *config.experiment_data_folder*.

    Returns
    -------
    pathlib.Path
        Fully-qualified path to the single matching TFRecord file.

    Raises
    ------
    FileNotFoundError
        If no file with the expected prefix exists.
    RuntimeError
        If more than one file matches the prefix (ambiguous).
    """
    split = split.lower()
    prefix = f"{split}_{units:02d}_{style_flag.upper()}_{smoothing_flag.upper()}"
    pattern = f"{prefix}*IEData.tfrecord.gz"
    matches = sorted(base_data_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No TFRecord starting with '{prefix}' in {base_data_dir}")
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous prefix '{prefix}': {matches}")
    return matches[0]


# ── Utility helpers ────────────────────────────────────────────────────────────

def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return the checkpoint with the highest *numeric* epoch identifier.

    Keras' ``ModelCheckpoint`` callback embeds the epoch number into the file
    name (e.g. ``epoch-012-val_*.keras``).  This helper extracts the number and
    returns the latest file so that training can resume from there.

    Parameters
    ----------
    ckpt_dir : pathlib.Path
        Directory that contains ``*.keras`` model files.

    Returns
    -------
    pathlib.Path | None
        Path to the checkpoint or ``None`` if the directory is empty.
    """
    epochs: Dict[int, Path] = {}
    for p in ckpt_dir.glob("*.keras"):
        m = re.search(r"epoch[-_](\d+)", p.name)
        if m:
            epochs[int(m.group(1))] = p
    return epochs[max(epochs)] if epochs else None


def _load_running_history(path: Path) -> Dict[str, List[float]]:
    """Load a *running* history JSON file if it exists else return an empty dict."""
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return defaultdict(list)


def _extend_history(store: Dict[str, List], fragment: Dict[str, List]):
    """Append values from *fragment* into the list-of-lists *store* in-place."""
    for k, v in fragment.items():
        store.setdefault(k, []).extend(v)


def _atomic_dump(obj: Dict, path: Path):
    """Write *obj* as JSON to *path* atomically (write-then-move)."""
    tmp = tempfile.NamedTemporaryFile("w", delete=False)
    try:
        json.dump(obj, tmp)
        tmp.close()
        shutil.move(tmp.name, path)
    finally:
        if Path(tmp.name).exists():
            Path(tmp.name).unlink(missing_ok=True)


# ── Main handler ───────────────────────────────────────────────────────────────

class ExperimentHandler:
    """Orchestrate a *multi-trial* experiment defined by a CSV specification.

    Each trial is trained for ``config.EPOCH_UNITS_PER_TRIAL`` epoch-units.  The
    handler creates a dedicated sub-folder per trial, handles resumption logic
    (checkpoint + ReduceLROnPlateau state + aggregated history) and finally
    stores a *final* model plus a JPEG learning-curve.
    """

    # ────────────────────────────────────────────────────────────────────
    def __init__(
        self,
        csv_path: Path | str,
        experiment_name: str,
        model_dir: Path | str = MODEL_DIR,
        data_folder: Path | str = experiment_data_folder,
        resume: bool = True,
        batch_size: int = PHYSICAL_BATCH_SIZE,
    ) -> None:
        """Construct a new *ExperimentHandler*.

        Parameters
        ----------
        csv_path : str | pathlib.Path
            Path to the CSV file whose rows specify the hyper-parameters for each
            trial (columns must match the expected headers).
        experiment_name : str
            Name of the experiment folder to be created inside ``MODEL_DIR``.
        model_dir : str | pathlib.Path, optional
            Root directory that will hold the *experiment* folder (default:
            ``config.MODEL_DIR``).
        data_folder : str | pathlib.Path, optional
            Directory that contains pre-built TFRecord datasets (default:
            ``config.experiment_data_folder``).
        resume : bool, optional
            If ``True`` (default) the handler will look for an existing
            checkpoint & aggregated history and continue training; otherwise it
            always starts fresh.
        batch_size : int, optional
            Mini-batch size fed to ``tf.data`` pipelines (default:
            ``config.PHYSICAL_BATCH_SIZE``).
        """
        self.csv_path = Path(csv_path)
        self.experiment = experiment_name
        self.batch_size = batch_size
        self.resume_flag = resume
        self.experiment_dir = Path(model_dir) / experiment_name
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.trials_df = pd.read_csv(self.csv_path)
        self.trials_df.columns = [c.strip() for c in self.trials_df.columns]
        self.base_data_dir = Path(data_folder) / Path("TestValTrain")

    # ────────────────────────────────────────────────────────────────────
    def run(self):
        """Iterate over CSV rows and execute all trials sequentially."""
        for _, row in self.trials_df.iterrows():
            trial_id = int(row["Trial"])
            trial_dir = self._prepare_trial_folder(trial_id)
            history_path = trial_dir / "history_full.json"
            running_hist = _load_running_history(history_path)
            start_unit = len(next(iter(running_hist.values()), []))

            # ── Resume / build model ────────────────────────────────
            ckpt_dir = trial_dir / "checkpoints"
            model = None
            if self.resume_flag and (ckpt := latest_checkpoint(ckpt_dir)):
                custom_objs = utils.get_custom_objects()
                model = models.load_model(ckpt, compile=False, custom_objects=custom_objs)
            if model is None:
                backend.clear_session(); gc.collect()
                model = self._build_model(row)
            else:                                   # resume
                loss_fn = self._select_loss(row)
                metrics = [
                    m(num_classes=5) if "num_classes" in m.__init__.__code__.co_varnames else m()
                    for m in Custom_Metrics.METRICS
                ]                    
                model.compile(optimizer=Custom_Optimizers.AccumOptimizer(accum_steps=cfg.ACCUM_STEPS),
                            loss=loss_fn,
                            metrics=metrics)

            # ── Loop over epoch-units ───────────────────────────────
            for unit in range(start_unit, EPOCH_UNITS_PER_TRIAL):
                print(f"\nTrial {trial_id:02d} • Epoch-Unit {unit+1}/{EPOCH_UNITS_PER_TRIAL}")

                train_ds, val_ds = self._load_datasets(row)
                cbs = self._make_callbacks(row, unit, trial_dir, model)

                history = model.fit(
                    train_ds,
                    validation_data=val_ds,
                    epochs=unit + 1,
                    initial_epoch=unit,
                    steps_per_epoch=STEPS_PER_EPOCH_UNIT,
                    callbacks=cbs,
                )

                _extend_history(running_hist, history.history)
                _atomic_dump(running_hist, history_path)

            # ── End-of-trial clean-up ───────────────────────────────
            Helper_Functions.plot_train_val_curve(
                history_object=type("H", (), {"history": running_hist})(),
                training_target_variable="no_background_auc",
            ).savefig(trial_dir / "train_val_curve.jpg")
            model.save(trial_dir / "final_model.keras")

        print("\n✅ All trials complete!")

    # ────────────────────────────────────────────────────────────────────
    def _prepare_trial_folder(self, num: int) -> Path:
        """Ensure the directory structure for a given trial exists.

        Creates ``Trial_XX/checkpoints`` (nested under the experiment folder)
        and returns the trial directory path.
        """
        tdir = self.experiment_dir / f"Trial_{num:02d}"
        (tdir / "checkpoints").mkdir(parents=True, exist_ok=True)
        return tdir

    # ---------------- Dataset helpers ----------------------------------
    def _dataset_path_from_row(self, row: pd.Series, split: str) -> Path:
        """Resolve the TFRecord file for *split* according to CSV flags."""
        units_flag = int(row["Size"])
        style_flag = str(row["Style"]).upper()[0]
        smoothing_flag = "C" if str(row["Smoothing"]).upper().startswith("C") else "B"
        return build_dataset_filepath(
            split,
            units_flag,
            style_flag,
            smoothing_flag,
            base_data_dir=self.base_data_dir,
        )

    def _load_datasets(self, row: pd.Series, test: bool = False):
        """Build ``tf.data`` datasets for training/validation.

        Parameters
        ----------
        row : pandas.Series
            The current CSV row describing the trial.
        test : bool, optional
            If ``True`` the *validation* dataset is taken from the ``test``
            TFRecord instead of ``val`` (useful for final evaluation).
        """
        train_path = self._dataset_path_from_row(row, "train")
        val_path = self._dataset_path_from_row(row, "test" if test else "val")
        train_ds = Data_Functions.prep_dataset_from_tfrecord(
            train_path,
            batch_size=self.batch_size,
            shuffled=True,
            cut_background=not bool(row["Background"]),
        ).repeat()
        val_ds = Data_Functions.prep_dataset_from_tfrecord(
            val_path,
            batch_size=self.batch_size,
            shuffled=False,
            cut_background=not bool(row["Background"]),
        )
        return train_ds, val_ds

    # ---------------- Model/loss building ------------------------------
    def _build_model(self, row: pd.Series):
        """Instantiate and compile a model based on CSV hyper‑parameters."""
                # Robustly convert the “Dilation” CSV column to a float.
        raw_dilation = row.get("Dilation", 1.0)
        if isinstance(raw_dilation, (int, float)):
            dilation_mult = float(raw_dilation)
        else:
            # Handle common non‑numeric tokens such as “Standard”, "None", or empty strings.
            try:
                dilation_mult = float(raw_dilation)
            except (TypeError, ValueError):
                dilation_mult = 1.0  # default

        raw_att = row.get("Attention", False)
        if isinstance(raw_att, str):
            use_attention = raw_att.strip().lower() in ("true", "1", "yes", "y")
        else:
            use_attention = bool(raw_att)
        
        # if we're cutting out the background channel, only emit 4 outputs
        background_removed = not bool(row.get("Background", False))      
        num_classes = 4 if background_removed else 5

        model = Custom_Models.create_modular_dcnn_model(
            dilation_multiplier=dilation_mult,
            use_local_attention=use_attention,
            use_long_range_attention=use_attention,
            use_final_attention=use_attention,
            num_classes=num_classes
        )
        loss_fn = self._select_loss(row)
        metrics = [copy.deepcopy(m) for m in Custom_Metrics.METRICS]
        model.compile(optimizer=Custom_Optimizers.AccumOptimizer(accum_steps=cfg.ACCUM_STEPS), loss=loss_fn, metrics=metrics)
        return model

    def _select_loss(self, row: pd.Series):
        """Return an appropriately parameterised loss object for the trial."""
        early_rewarding = bool(row["Early Rewarding"])
        smoothing = str(row["Smoothing"]).lower()
        background_removed = not bool(row["Background"])

        kwargs = dict(
            dominant_class_index=cfg.DOMINANT_CLASS_INDEX,
            dominant_correct_multiplier=cfg.DOMINANT_CORRECT_MULTIPLIER,
            dominant_incorrect_multiplier=cfg.DOMINANT_INCORRECT_MULTIPLIER,
            other_class_true_positive_multiplier=cfg.OTHER_TP_MULTIPLIER,
            other_class_false_negative_multiplier=cfg.OTHER_FN_MULTIPLIER,
            other_class_false_positive_multiplier=cfg.OTHER_FP_MULTIPLIER,
            other_class_true_negative_multiplier=cfg.OTHER_TN_MULTIPLIER,
            background_removed=background_removed,
            threshold=cfg.THRESHOLD,
            focal_gamma=cfg.FOCAL_GAMMA,
            focal_alpha=cfg.FOCAL_ALPHA,
        )

        if smoothing == "custom":
            BaseLoss = Custom_Losses.CustomBinaryFocalLoss
            kwargs.update(
                smoothing_multiplier=cfg.INCORRECT_SMOOTHING_MULTIPLIER,
                smoothing_as_correct=cfg.DEFAULT_SMOOTHING_AS_CORRECT,
            )
        else:
            BaseLoss = Custom_Losses.AllBinaryFocalLoss
            # Remove 'threshold' since AllBinaryFocalLoss loss doesn't use it
            # Threshold strictly required for custom "smoothing" comparison 
            # to evaluate whether to reward or punish a guess in the loss calculation
            kwargs.pop("threshold", None)
            kwargs.update(label_smoothing=cfg.LABEL_SMOOTHING if smoothing == "proper" else 0)

        if early_rewarding:
            kwargs["swap_epoch"] = cfg.SWAP_EPOCH
            LossClass = (
                Custom_Losses.SwitchingFocalLoss
                if BaseLoss is Custom_Losses.CustomBinaryFocalLoss
                else Custom_Losses.SwitchingBinaryCrossentropyLoss
            )
        else:
            LossClass = BaseLoss
        return LossClass(**kwargs)

    # ---------------- Callbacks ----------------------------------------
    def _make_callbacks(
        self,
        row: pd.Series,
        unit_idx: int,
        trial_dir: Path,
        model,
    ) -> List[callbacks.Callback]:
        """Assemble a fresh list of callbacks for the *current* epoch-unit."""
        # Start with a shallow copy so that the global list is never mutated.
        cbs = [copy.deepcopy(cb) for cb in Custom_Callbacks.CALLBACKS]

        # --- TensorBoard ------------------------------------------------
        tb_dir = LOG_DIR / self.experiment / f"trial_{row['Trial']:02d}" / f"unit_{unit_idx:02d}"
        cbs.append(callbacks.TensorBoard(log_dir=tb_dir, histogram_freq=1))

        # --- Epoch updater for switchable loss --------------------------
        if hasattr(model.loss, "epoch_var"):
            cbs.append(Custom_Losses.EpochUpdater(model.loss))

        # --- LR-scheduler state path (unique per trial) -----------------
        lr_state_dir = trial_dir / LR_STATE_SAVE_SUBDIR
        lr_state_dir.mkdir(parents=True, exist_ok=True) 
        lr_state_path = trial_dir / LR_STATE_SAVE_SUBDIR / LR_STATE_SAVE_FILENAME
        for cb in cbs:
            if isinstance(cb, callbacks.ModelCheckpoint):
                cb.filepath = str(trial_dir / cfg.CHECKPOINT_SUBDIR / cfg.CHECKPOINT_FILENAME)
            if isinstance(cb, Custom_Callbacks.StatefulReduceLROnPlateau):
                cb.state_save_filepath = lr_state_path
                if lr_state_path.exists() and self.resume_flag and unit_idx == 0:
                    cb.load_state_from_file(lr_state_path)
        return cbs


# ── CLI helper ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run an Intron-Exon experiment from a CSV spec.")
    parser.add_argument("csv", help="Path to experiment CSV file")
    parser.add_argument("name", help="Experiment folder name (under Models/)")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoints/history")
    parser.add_argument("--batch", type=int, default=PHYSICAL_BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    handler = ExperimentHandler(
        csv_path=args.csv,
        experiment_name=args.name,
        resume=not args.no_resume,
        batch_size=args.batch,
    )
    handler.run()