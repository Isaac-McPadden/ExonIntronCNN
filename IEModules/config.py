# config.py
import os
from . import Custom_Metrics as cm

# Project base directory (absolute path)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR     = os.path.abspath(os.path.join(BASE_DIR, "../Datasets"))
LOG_DIR      = os.path.abspath(os.path.join(BASE_DIR, "../Logs"))
MODEL_DIR    = os.path.abspath(os.path.join(BASE_DIR, "../Models"))
MODULE_DIR   = BASE_DIR
NOTEBOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Notebooks"))

# ── Constants ──────────────────────────────────────────────────────────────────
EPOCH_UNIT_SIZE = int(1182630/5)
EPOCH_UNITS_PER_TRIAL = 10          # <- Each trial always runs exactly 10 units
DEFAULT_BATCH_SIZE = 28
STEPS_PER_EPOCH_UNIT = int(EPOCH_UNIT_SIZE/DEFAULT_BATCH_SIZE) # Becomes steps_per_epoch


METRICS = [
    cm.CustomNoBackgroundAUC,
    cm.CustomNoBackgroundF1Score,
    cm.CustomFalsePositiveDistance,
    cm.CustomBackgroundOnlyF1Score,
    cm.CustomNoBackgroundAccuracy,
    cm.CustomNoBackgroundPrecision,
    cm.CustomNoBackgroundRecall,
    cm.CustomConditionalF1Score
]



if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("LOG_DIR:", LOG_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("NOTEBOOKS_DIR:", NOTEBOOK_DIR)