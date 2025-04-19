import time
import sys
import os
import glob
import math
import threading
import concurrent.futures as cf
import random
import re
import gc

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, metrics, losses, callbacks, optimizers, models, utils
from keras import backend as K
import keras_tuner as kt
from pyfaidx import Fasta

from . import Custom_Callbacks
from . import Custom_Losses
from . import Custom_Metrics
from . import Custom_Models
from . import Data_Functions
from . import Genetic_Data_Pipeline
from . import Helper_Functions
from .config import DATA_DIR, LOG_DIR, MODEL_DIR, MODULE_DIR, NOTEBOOK_DIR
'''
Run the code below in notebooks to get them to add modules to the path
'''
# Adjust the path based on file’s relative location to project root folder
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# Then import IEModules as iem