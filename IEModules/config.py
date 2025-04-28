# config.py
import sys
from pathlib import Path
from . import Custom_Metrics as cm
from . import Custom_Callbacks as cc
from keras import optimizers

# ── Project Directories ──────────────────────────────────────────────────────────────────
# Project base directory (absolute path)
# file currently lives in …/IEModules/config.py
MODULE_DIR = Path(__file__).parent
PROJECT_ROOT = MODULE_DIR.parent

DATA_DIR     = PROJECT_ROOT / "Datasets"
LOG_DIR      = PROJECT_ROOT / "Logs"
MODEL_DIR    = PROJECT_ROOT / "Models"
NOTEBOOK_DIR = PROJECT_ROOT / "Notebooks"

'''
MANUALLY CHANGE THIS EVERY EXPERIMENT
'''
experiment_data_folder = DATA_DIR / "Experiment 01"
experiment_folder = MODEL_DIR / "Experiment 01"


# ── Constants ──────────────────────────────────────────────────────────────────
EPOCH_UNIT_SIZE = int(1182630/5)
EPOCH_UNITS_PER_TRIAL = 10          # <- Each trial always runs exactly 10 units
DEFAULT_BATCH_SIZE = 28
STEPS_PER_EPOCH_UNIT = int(EPOCH_UNIT_SIZE/DEFAULT_BATCH_SIZE) # Becomes steps_per_epoch
SEED = 42

# ── Model Parameters ──────────────────────────────────────────────────────────────────
# Have to set this up in the experiment handler as they change every trial
# input_dim=5,
# sequence_length=5000,
# num_classes=5,
# use_local_attention=True,
# use_long_range_attention=True,
# use_final_attention=True,
# dilation_multiplier=1.0 

# ── Optimizer settings ──────────────────────────────────────────────────────────────────
INITIAL_LEARNING_RATE = 0.001
INITIAL_OPTIMIZER = optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE)
GENERIC_OPTIMIZER = optimizers.Adam
# Plateau rate reduction handled by callback

'''BREADCRUMB FOR SELF: Still need loss config
More importantly, need to deal with lr_state_dir and path issues for the json save
'''

# ── Callback checkpoints config ──────────────────────────────────────────────────────────────────
CHECKPOINT_SUBDIR            = "Checkpoints"
LR_STATE_SAVE_PATH         = "LR_State/reduce_lr_state.json"
# filename template with Keras epoch & metric tokens
CHECKPOINT_FILENAME          = "epoch-{epoch:03d}-val_no_background_auc-{val_no_background_auc:.4f}.keras"
CHECKPOINT_MONITOR           = "val_no_background_auc"
CHECKPOINT_MODE              = "max"  # change to min for val_loss if using that for some reason
CHECKPOINT_SAVE_BEST_ONLY    = False  # save model always 
CHECKPOINT_SAVE_WEIGHTS_ONLY = False  # save full model (architecture + weights)
CHECKPOINT_SAVE_FREQ         = "epoch"

# ── Callbacks ──────────────────────────────────────────────────────────────────
REDUCE_LR_CALLBACK = cc.reduce_lr_cb
CALLBACKS = [
    cc.cleanup_cb,
    cc.checkpoint_cb,
    cc.early_stopping_cb,
    REDUCE_LR_CALLBACK,
    ]

# ── Metrics ──────────────────────────────────────────────────────────────────
METRICS = [
    cm.CustomNoBackgroundAUC, # PR-AUC, most important metric
    cm.CustomNoBackgroundF1Score,
    # cm.CustomFalsePositiveDistance, # Not actually that useful
    # cm.CustomBackgroundOnlyF1Score, # Definitely not useful
    cm.CustomNoBackgroundAccuracy, # For kicks
    cm.CustomNoBackgroundPrecision,
    cm.CustomNoBackgroundRecall
]

# ── Loss kwargs config ──────────────────────────────────────────────────────────────────


def main():
    # grab this module object
    this_mod = sys.modules[__name__]
    # iterate through all names in the module namespace
    for name, val in vars(this_mod).items():
        print(f"{name} = {val}")

if __name__ == "__main__":
    main()