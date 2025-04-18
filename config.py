# config.py
import os

# Project base directory (absolute path)
BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Subdirectories
DATA_DIR = os.path.join(BASE_DIR, "Datasets")
LOG_DIR = os.path.join(BASE_DIR, "Logs")
MODEL_DIR = os.path.join(BASE_DIR, "Models")
MODULE_DIR = os.path.join(BASE_DIR, "IEModules")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "Notebooks") # Probably unnecessary