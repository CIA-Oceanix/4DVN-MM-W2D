
print('###################################')
print('Experiment : ')
print('###################################')

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'utls'))
sys.path.append(os.path.join(os.getcwd(), '4dvar'))

from dotenv import load_dotenv
import json
import numpy as np

from pathmng import PathManager
from dlmodels import LitModel, Phi_r
from dutls import W2DDataModule


load_dotenv(os.path.join(os.getcwd(), 'config.env'))
with open('./cparams.json', 'r') as f:
    CPARAMS = json.load(f)
f.close()

PATH_DATA   = os.getenv('PATH_DATA')
PATH_MODEL  = os.getenv('PATH_MODEL')
BATCH_SIZE  = CPARAMS['BATCH_SIZE']

def main():
    
    # initialize path manager
    pm = PathManager(PATH_MODEL, '4DVN-W2D', versioning = True)
    
    # initialize datamodule
    ds = W2DDataModule(PATH_DATA, BATCH_SIZE)
    
    # initialize and configure models
    Phi = Phi_r()
    lit_model = LitModel(Phi)
    
    # configure trainer properties, trainer
    
    # train / test
    
    # print report in the proper target directory
    pm.save_configfiles(CPARAMS, 'config_params')
    
    perf_dict = {
        'mse' : np.random.normal(0, 1),
        'kld' : np.random.normal(0, 1)
    }
    pm.print_evalreport(perf_dict)
    
    pass
#end
    

if __name__ == '__main__':
    
    main()
#end
