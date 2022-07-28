import os
from sys import getsizeof
import numpy as np
import pandas as pd
from tqdm import tqdm
import netCDF4 as nc
import argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

PATH_DATA  = os.getenv('PATH_DATA')

parser = argparse.ArgumentParser()
parser.add_argument('-c', type = int, default = 1)
parser.add_argument('-f', type = int, default = 1)
args = parser.parse_args()

print(args)


def distance_km_from_lat_lon(lat1, lat2, lon1, lon2):
    ''' Matches with https://www.nhc.noaa.gov/gccalc.shtml '''
    
    earth_radius = 6371
    diff_latitude = np.radians(lat1 - lat2)
    diff_longitude = np.radians(lon1 - lon2)
    a = np.sin(diff_latitude / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(diff_longitude / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = earth_radius * c
    return d
#end

def size_obj(obj, unit = 'MiB'):
    
    if unit == 'KiB':
        return getsizeof(obj) / (2**10)
    elif unit == 'MiB':
        return getsizeof(obj) / (2**20)
    elif unit == 'GiB':
        return getsizeof(obj) / (2**30)
    elif unit == 'kB':
        return getsizeof(obj) / (1e3)
    elif unit == 'MB':
        return getsizeof(obj) / (1e5)
    elif unit == 'GB':
        return getsizeof(obj) / (1e8)
    else:
        raise NotImplementedError('please MiB')
    #end
#end

# --------------------------- #
# REFERENCE PERIOD            #
# 01/01/2019 — 01/01/2021     #
# --------------------------- #

day_start, month_start, year_start = 1, 1, 2019
day_end,   month_end,   year_end   = 1, 1, 2021

date_end   = pd.to_datetime(f'{year_end}-{month_end}-{day_end} 23:00:00', format = '%Y-%m-%d %H:%M:%S')
date_start = pd.to_datetime(f'{year_start}-{month_start}-{day_start} 00:00:00', format = '%Y-%m-%d %H:%M:%S')

timedelta = pd.Timedelta(date_end - date_start)
print(timedelta)
delta_days = timedelta.days
delta_hours = np.int32(timedelta / np.timedelta64(1, 'h'))
print('Hours since {} to {} : {}'.format(date_start, date_end, delta_hours+1))

XLAT_MIN   = 289
XLAT_MAX   = 323
XLAT_STEP  = 1
XLONG_MIN  = 289
XLONG_MAX  = 323
XLONG_STEP = 1
TIME_MIN   = 0
TIME_MAX   = delta_hours + 1
TIME_STEP  = 1
#end

URL = "https://tds.marine.rutgers.edu/thredds/dodsC/cool/ruwrf/wrf_4_1_3km_processed/WRF_4.1_3km_Processed_Dataset_Best?"\
    + "XLONG[{}:{}:{}][{}:{}:{}],".format(XLONG_MIN, XLONG_STEP, XLONG_MAX, XLONG_MIN, XLONG_STEP, XLONG_MAX) \
    + "XLAT[{}:{}:{}][{}:{}:{}],".format(XLAT_MIN, XLAT_STEP, XLAT_MAX, XLAT_MIN, XLAT_STEP, XLONG_MAX) \
    + "time[{}:{}:{}],".format(TIME_MIN, TIME_STEP, TIME_MAX)\
    + "U10[{}:{}:{}][{}:{}:{}][{}:{}:{}],".format(TIME_MIN, TIME_STEP, TIME_MAX,
                                                  XLONG_MIN, XLONG_STEP, XLONG_MAX,
                                                  XLAT_MIN, XLAT_STEP, XLAT_MAX) \
    + "V10[{}:{}:{}][{}:{}:{}][{}:{}:{}]".format(TIME_MIN, TIME_STEP, TIME_MAX,
                                               XLONG_MIN, XLONG_STEP, XLONG_MAX,
                                               XLAT_MIN, XLAT_STEP, XLAT_MAX)

dataset = nc.Dataset(URL)
lat     = dataset['XLAT']
lon     = dataset['XLONG']
times   = dataset['time']

size = size_obj(dataset['U10'], unit = 'KiB')
print(f'Size of u10 : {size} KiB')
size = size_obj(dataset['V10'], unit = 'KiB')
print(f'Size of v10 : {size} KiB')

unit = 'KiB'

print('------ DIMENSIONS ------\n')
for dim in dataset.dimensions.values():
    print(dim)
#end

print()
print('------ VARIABLES -------\n')
for var in dataset.variables.values():
    print(var)
    size = size_obj(var, unit = unit)
    print(f'SIZE in {unit} : {size}')
    print()
#end

DS_CREATE = args.c
DS_FETCH  = args.f

U10 = np.zeros((TIME_MAX, XLONG_MAX - XLONG_MIN + 1, XLAT_MAX - XLAT_MIN + 1))
V10 = np.zeros((TIME_MAX, XLONG_MAX - XLONG_MIN + 1, XLAT_MAX - XLAT_MIN + 1))

if DS_CREATE == 1:
    
    for i in tqdm(range(10)):
        
        U10[i,:,:] = dataset['U10'][i,:,:]
        V10[i,:,:] = dataset['V10'][i,:,:]
    #end
    
    wind = np.sqrt((U10**2 + V10**2))
    print('Scalar wind shape : ', wind.shape)
    print('Size of scalar wind in MiB : ', size_obj(wind, unit = 'MiB'))
    
    with open(os.path.join(PATH_DATA, 'patch_modwind2D_24h.npy'), 'wb') as f:
        np.save(f, wind, allow_pickle = True)
    f.close()
#end

if DS_FETCH:
    
    pass
#end

# if DS_CREATE == 1:
    
#     for i in tqdm(range(delta_hours+1)):
        
#         U10 = dataset['U10'][i,:,:]
#         V10 = dataset['V10'][i,:,:]
        
#         with open(os.path.join(PATH_DATA, f'u10_run{i}.npy'), 'wb') as f:
#             np.save(f, np.array(U10))
#         f.close()
        
#         with open(os.path.join(PATH_DATA, f'v10_run{i}.npy'), 'wb') as f:
#             np.save(f, np.array(V10))
#         f.close()
#     #end
    
# if DS_FETCH == 1:
    
#     U10 = np.zeros((TIME_MAX, XLONG_MAX - XLONG_MIN + 1, XLAT_MAX - XLAT_MIN + 1))
#     V10 = np.zeros((TIME_MAX, XLONG_MAX - XLONG_MIN + 1, XLAT_MAX - XLAT_MIN + 1))
    
#     for i in tqdm(range(delta_hours+1)):
        
#         U10[i,:,:] = np.load(open(os.path.join(PATH_DATA, f'u10_run{i}.npy'), 'rb'))
#         V10[i,:,:] = np.load(open(os.path.join(PATH_DATA, f'v10_run{i}.npy'), 'rb'))
#     #end
    
#     wind = np.sqrt((U10**2 + V10**2))
#     print('Scalar wind shape : ', wind.shape)
#     print('Size of scalar wind in MiB : ', size_obj(wind, unit = 'MiB'))
    
#     with open(os.path.join(PATH_DATA, 'patch_modwind2D_24h.npy'), 'wb') as f:
#         np.save(f, wind, allow_pickle = True)
#     f.close()
# #end
        

lat   = np.array(lat)
lon   = np.array(lon)
times = np.array(times)

lon_min = lon.min()
lon_max = lon.max()
lat_min = lat.min()
lat_max = lat.max()

print(f'Longitude range [ {lon_min:.4f} : {lon_max:.4f} ] — shape : {lon.shape}')
print(f'Latitude  range [ {lat_min:.4f} : {lat_max:.4f} ] — shape : {lat.shape}')
print(f'Time range [ {times.min()} : {times.max()} ] — shape : {times.shape}')
print(f'u10 , v10 shapes : {U10.shape} , {V10.shape}')

sizeu, sizev = size_obj(U10, unit = 'KiB'), size_obj(V10, unit = 'KiB')
print(f'Velocity field u10, v10 memory space : {sizeu:.4f}, {sizev:.4f} KiB')
sizeu, sizev = size_obj(U10, unit = 'MiB'), size_obj(V10, unit = 'MiB')
print(f'Velocity field u10, v10 memory space : {sizeu:.4f}, {sizev:.4f} MiB')


width  = distance_km_from_lat_lon(lat[0,0], lat[0,-1], lon[0,0], lon[0,-1])
height = distance_km_from_lat_lon(lat[0,0], lat[-1,0], lon[0,0], lon[-1,0])

diff_deg_lat = np.zeros((lat.shape[0]-1, lat.shape[1]))
diff_deg_lon = np.zeros((lon.shape[0],   lon.shape[1]-1))
diff_km_lat  = np.zeros((lat.shape[0]-1, lat.shape[1]))
diff_km_lon  = np.zeros((lon.shape[0],   lon.shape[1]-1))

for i in range(lat.shape[0]-1):
    diff_deg_lat[i,:] = np.abs(lat[i,:] - lat[i+1,:])
    diff_km_lat[i,:]  = distance_km_from_lat_lon( lat[i,:], lat[i+1,:], lon[i,:], lon[i+1,:] )
#end
for i in range(lon.shape[1]-1):
    diff_deg_lon[:,i] = np.abs(lon[:,i] - lon[:,i+1])
    diff_km_lon[:,i]  = distance_km_from_lat_lon( lat[:,i], lat[:,i+1], lon[:,i], lon[:,i+1] )
#end

print(f'Width  ; resolution [km] = {width:.2f} ; {width/lat.shape[1]:.2f} or {diff_km_lat.mean():.2f} ± {diff_deg_lat.std():.2f} ; Average resolution [°] = {diff_deg_lat.mean():.2f} ± {diff_deg_lat.std():.2f}')
print(f'Height ; resolution [km] = {height:.2f} ; {height/lon.shape[0]:.2f} or {diff_km_lon.mean():.2f} ± {diff_deg_lon.std():.2f} ; Average resolution [°] = {diff_deg_lon.mean():.2f} ± {diff_deg_lon.std():.2f}')

