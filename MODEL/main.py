
print('###################################')
print('Experiment : ')
print('###################################')

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'utls'))
sys.path.append(os.path.join(os.getcwd(), '4dvar'))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

import json
import numpy as np
import torch

from pathmng import PathManager
from dlmodels import LitModel


if __name__ == '__main__':
    
    PATH_DATA   = os.getenv('PATH_DATA')
    PATH_MODEL  = os.getenv('PATH_MODEL')

    with open(os.path.join(os.getcwd(), 'cparams.json'), 'r') as filestream:
        CPARAMS = json.load(filestream)
    filestream.close()
    
    pm = PathManager(PATH_MODEL, '4DVN-W2D', versioning = False)