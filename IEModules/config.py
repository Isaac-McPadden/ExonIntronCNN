# config.py
import os

# Project base directory (absolute path)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

DATA_DIR     = os.path.abspath(os.path.join(BASE_DIR, "../Datasets"))
LOG_DIR      = os.path.abspath(os.path.join(BASE_DIR, "../Logs"))
MODEL_DIR    = os.path.abspath(os.path.join(BASE_DIR, "../Models"))
MODULE_DIR   = BASE_DIR
NOTEBOOK_DIR = os.path.abspath(os.path.join(BASE_DIR, "../Notebooks"))

if __name__ == "__main__":
    print("BASE_DIR:", BASE_DIR)
    print("DATA_DIR:", DATA_DIR)
    print("LOG_DIR:", LOG_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    print("NOTEBOOKS_DIR:", NOTEBOOK_DIR)