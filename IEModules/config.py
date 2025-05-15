# config.py
import sys
from pathlib import Path
# import Custom_Metrics as cm
# import Custom_Callbacks as cc
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
Default Vars
'''
experiment_data_folder = DATA_DIR / "Experiment 01"
experiment_folder = MODEL_DIR / "Experiment 01"
target_training_variable = "no_background_auc"


# ── Constants ──────────────────────────────────────────────────────────────────
EPOCH_UNIT_SIZE = int(0.8*1182630/5) # The approximate number of samples in a single unaugmented dataset
EPOCH_UNITS_PER_TRIAL = 10          # <- Each trial always runs exactly 10 units
DEFAULT_BATCH_SIZE = 28
# STEPS_PER_EPOCH_UNIT = int(EPOCH_UNIT_SIZE/DEFAULT_BATCH_SIZE) # Becomes steps_per_epoch
SEED = 42
PHYSICAL_BATCH_SIZE = 2
ACCUM_STEPS = DEFAULT_BATCH_SIZE // PHYSICAL_BATCH_SIZE
STEPS_PER_EPOCH_UNIT = int(EPOCH_UNIT_SIZE // PHYSICAL_BATCH_SIZE)

# ── Test Constants ──────────────────────────────────────────────────────────────────
# EPOCH_UNIT_SIZE = 100
# EPOCH_UNITS_PER_TRIAL = 10          # <- Each trial always runs exactly 10 units
# DEFAULT_BATCH_SIZE = 20
# # STEPS_PER_EPOCH_UNIT = int(EPOCH_UNIT_SIZE/DEFAULT_BATCH_SIZE) # Becomes steps_per_epoch
# SEED = 42
# PHYSICAL_BATCH_SIZE = 2
# ACCUM_STEPS = DEFAULT_BATCH_SIZE // PHYSICAL_BATCH_SIZE
# STEPS_PER_EPOCH_UNIT = int(EPOCH_UNIT_SIZE // PHYSICAL_BATCH_SIZE)
# '''
# Default Vars
# '''
# experiment_data_folder = DATA_DIR / "PROGRAM TEST 01"
# experiment_folder = MODEL_DIR / "PROGRAM TEST 01"
# target_training_variable = "no_background_auc"

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
GENERIC_OPTIMIZER = optimizers.Adam()
# Plateau rate reduction handled by callback

# ── Callback checkpoints config ──────────────────────────────────────────────────────────────────
CHECKPOINT_SUBDIR            = "Checkpoints"
LR_STATE_SAVE_SUBDIR         = "LR_State"
LR_STATE_SAVE_FILENAME       = "reduce_lr_state.json"
# filename template with Keras epoch & metric tokens
PREVAL_CHECKPOINT_MONITOR    = "no_background_auc"
PREVAL_CHECKPOINT_FILENAME   = "epoch-{epoch:03d}-preval-chkpt.keras"
CHECKPOINT_FILENAME          = "epoch-{epoch:03d}-val_no_background_auc-{val_no_background_auc:.4f}.keras"
CHECKPOINT_MONITOR           = "val_no_background_auc"
CHECKPOINT_MODE              = "max"  # change to min for val_loss if using that for some reason
CHECKPOINT_SAVE_BEST_ONLY    = False  # save model always 
CHECKPOINT_SAVE_WEIGHTS_ONLY = False  # save full model (architecture + weights)
CHECKPOINT_SAVE_FREQ         = "epoch"

# ── Loss kwargs config ──────────────────────────────────────────────────────────────────
DOMINANT_CLASS_INDEX = 0              # Never needs to be changed really
DOMINANT_CORRECT_MULTIPLIER = 0.99    # Reward when dominant class is correct
DOMINANT_INCORRECT_MULTIPLIER = 2.5   # Penalty when dominant class is incorrect
OTHER_TP_MULTIPLIER = 0.05            # Reward when y_true==1 and prediction is positive
OTHER_FN_MULTIPLIER = 3.0             # Punish when y_true==1 but prediction is negative
OTHER_FP_MULTIPLIER = 1.0             # Punish when y_true==0 but prediction is positive
OTHER_TN_MULTIPLIER = 0.99            # Reward when y_true==0 and prediction is negative
THRESHOLD = 0.5                       # Threshold to decide if a prediction is "positive"
FOCAL_GAMMA = 2.5                     # Focusing parameter gamma
FOCAL_ALPHA = 0.25                    # Balance parameter alpha
INCORRECT_SMOOTHING_MULTIPLIER = 1.1  # For smooothing as correct is False, Scales the effect of a custom smoothed label
CORRECT_SMOOTHING_MULTIPLIER = 0.5    # For smoothing as correct is True, Scales the effect of a custom smoothed label
DEFAULT_SMOOTHING_AS_CORRECT = False  # If True, a high prediction on a smoothed label is rewarded; else, punished
LABEL_SMOOTHING = 0.02                # Proper label smoothing value.  
SWAP_EPOCH = 3                        # Number of rewarding epochs


def main():
    # grab this module object
    this_mod = sys.modules[__name__]
    # iterate through all names in the module namespace
    for name, val in vars(this_mod).items():
        print(f"{name} = {val}")

if __name__ == "__main__":
    main()

"""   
Todos:
    Potentially, latest_checkpoint() sorts lexicographically could be fixed
    Path(str) instead of passing strings to things that use paths.
    Clear Keras session between trials to release GPU memory
    Temp vars in config should be made into cli args
    LR_state needs per trial json
"""