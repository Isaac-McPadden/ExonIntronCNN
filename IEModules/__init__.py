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

from IEModules import Custom_Callbacks
from IEModules import Custom_Losses
from IEModules import Custom_Metrics
from IEModules import Custom_Models
from IEModules import Data_Functions
from IEModules import Genetic_Data_Pipeline
from IEModules import Helper_Functions
from IEModules import config
'''
Run the code below in notebooks to get them to add modules to the path
'''
# Adjust the path based on file’s relative location to project root folder
# project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# Then import IEModules as iem