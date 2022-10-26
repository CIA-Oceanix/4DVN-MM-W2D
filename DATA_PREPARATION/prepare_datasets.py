
import numpy as np
import os
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

PATH_DATA = os.getenv('PATH_DATA')
PATH_DATA = os.path.join(PATH_DATA, 'MABay', 'winds_24h')

parser = argparse.ArgumentParser()
parser.add_argument('-case', type = str)
parser.add_argument('-pext', type = int)
args = parser.parse_args()

case = args.case
pext = args.pext

if case is None:
    case = 'melee'
    pext = 150
#end

wind = np.load(os.path.join(PATH_DATA, 'patch_modw_01012019-31122020.npy'))

if case == 'coast':
    _wind = wind[:, :pext, :pext]
elif case == 'sea':
    _wind = wind[:, :pext, -pext:]
elif case == 'melee':
    wind_lr = wind[:, :pext, :pext]
    wind_hr = wind[:, :pext, -pext:]
    _wind = np.concatenate((wind_lr, wind_hr), axis = 2)
else:
    raise ValueError('Not implemented !!!')
#ends

np.save(os.path.join(PATH_DATA, f'ds_{case}_{pext}px.npy'), _wind)
