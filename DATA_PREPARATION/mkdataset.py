import os
from sys import getsizeof
import numpy as np
import pandas as pd
from tqdm import tqdm
from global_land_mask import globe
import netCDF4 as nc
import argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))

PATH_DATA = os.getenv('PATH_DATA')

parser = argparse.ArgumentParser()
parser.add_argument('-create',     type = int, default = 1)
parser.add_argument('-fetch',      type = int, default = 1)
parser.add_argument('-dscomplete', type = int, default = 0)
parser.add_argument('-crop',       type = int, default = 0)
parser.add_argument('-dinit',      type = int)
parser.add_argument('-minit',      type = int)
parser.add_argument('-yinit',      type = int)
parser.add_argument('-dend',       type = int)
parser.add_argument('-mend',       type = int)
parser.add_argument('-yend',       type = int)
args = parser.parse_args()

if args.create > 1:
    args.create = 1
if args.fetch > 1:
    args.fetch = 1
if args.dscomplete > 1:
    args.dscomplete = 1
#end

if args.crop == 0:
    _CROP_IMG = None
else:
    _CROP_IMG = args.crop
#end


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

def save_netCDF4_dataset(lat, lon, time, mask, wind, indices, ds_name, day_start, month_start, year_start):
    
    if os.path.exists(os.path.join(PATH_DATA, ds_name)):
        os.remove(os.path.join(PATH_DATA, ds_name))
        print('Old Dataset removed ...')
    #end
    
    print('Creating new netCDF4 Dataset ...')
    nc_dataset = nc.Dataset(os.path.join(PATH_DATA, '{}.nc'.format(ds_name)), mode = 'w', format = 'NETCDF4_CLASSIC')
    nc_dataset.createDimension('south-north', lat.shape[0])
    nc_dataset.createDimension('west-east', lon.shape[1])
    nc_dataset.createDimension('time', time.__len__())
    
    nc_lat = nc_dataset.createVariable('lat', np.float32, ('south-north', 'west-east'))
    nc_lat.units = 'degree_north'
    nc_lat.long_name = 'latitude'
    nc_lon = nc_dataset.createVariable('lon', np.float32, ('south-north', 'west-east'))
    nc_lon.units = 'degree_east'
    nc_lon.long_name = 'longitude'
    nc_mask = nc_dataset.createVariable('mask_land', np.float32, ('south-north', 'west-east'))
    nc_mask.units = 'm'
    nc_mask.long_name = 'Mask_land_sea'
    nc_time = nc_dataset.createVariable('time', np.float64, ('time',))
    nc_time.units = f'hours_since_{day_start:02d}/{month_start:02d}/{year_start}'
    nc_time.long_name = 'hours'
    nc_wind = nc_dataset.createVariable('wind', np.float32, ('time', 'south-north', 'west-east'))
    nc_wind.units = 'm s-1'
    nc_wind.long_name = 'model_wind'
    nc_windices = nc_dataset.createVariable('indices', np.int32, ('time',))
    nc_windices.units = 'none'
    nc_windices.long_name = 'indices_of_wind_images'
    
    nc_lat[:,:] = lat
    nc_lon[:,:] = lon
    nc_time[:] = time
    nc_mask[:,:] = mask
    nc_wind[:,:,:] = wind
    nc_windices = indices
    
    print('Dataset save. Closing ...')
    nc_dataset.close()
    print('Dataset close.')
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
        return getsizeof(obj) / (1e6)
    elif unit == 'GB':
        return getsizeof(obj) / (1e9)
    else:
        raise NotImplementedError('please MiB')
    #end
#end

# --------------------------- #
# REFERENCE PERIOD            #
# 01/01/2019 — 01/01/2021     #
# --------------------------- #

day_start, month_start, year_start = args.dinit, args.minit, args.yinit
day_end,   month_end,   year_end   = args.dend, args.mend, args.yend

date_end   = pd.to_datetime(f'{year_end}-{month_end}-{day_end} 23:00:00', format = '%Y-%m-%d %H:%M:%S')
date_start = pd.to_datetime(f'{year_start}-{month_start}-{day_start} 00:00:00', format = '%Y-%m-%d %H:%M:%S')

timedelta = pd.Timedelta(date_end - date_start)
print(timedelta)
delta_days = timedelta.days
delta_hours = np.int32(timedelta / np.timedelta64(1, 'h'))
print('Hours since {} to {} : {}'.format(date_start, date_end, delta_hours+1))

time_range = pd.date_range(start = date_start, end = date_end, freq = 'h').tz_localize('UTC')
print('Total hours with date_range : {}'.format(time_range.__len__()))

indices = np.arange(time_range.__len__())

XLAT_MIN   = 0
XLAT_MAX   = 323
XLAT_STEP  = 1
XLONG_MIN  = 0
XLONG_MAX  = 323
XLONG_STEP = 1
TIME_MIN   = 0
TIME_MAX   = delta_hours + 1
TIME_STEP  = 1
CROP_IMG   = _CROP_IMG


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

DS_CREATE   = args.create
DS_FETCH    = args.fetch
DS_COMPLETE = args.dscomplete

lat   = np.array(lat)
lon   = np.array(lon)
times = np.array(times)

lon_min = lon.min()
lon_max = lon.max()
lat_min = lat.min()
lat_max = lat.max()

if np.bool_(DS_CREATE):
    
    for i in tqdm(range(delta_hours)):
        
        _u10 = dataset['U10'][i,:,:].data
        _v10 = dataset['V10'][i,:,:].data
        
        w = np.sqrt((_u10**2 + _v10**2))
        with open(os.path.join(PATH_DATA, 'w{}.npy'.format(i)), 'wb') as f:
            np.save(f, w, allow_pickle = True)
        f.close()
    #end
    
    print('***FILES SAVED SUCCESS***')
#end

if np.bool_(DS_FETCH):
    
    wind = np.zeros((TIME_MAX, XLONG_MAX - XLONG_MIN + 1, XLAT_MAX - XLAT_MIN + 1))
    filename = f'patch_modw_{day_start:02d}{month_start:02d}{year_start}-{day_end:02d}{month_end:02d}{year_end}.npy'
    
    if DS_COMPLETE:
        
        with open(os.path.join(PATH_DATA, filename), 'rb') as f:
            wind = np.load(f)
        #end
        
        print('***FILES IMPORTED SUCCESS***')
    else:
        for i in tqdm(range(delta_hours)):
            
            with open(os.path.join(PATH_DATA, 'w{}.npy'.format(i)), 'rb') as f:
                wind[i,:,:] = np.load(f)
            f.close()
        #end
        
        with open(os.path.join(PATH_DATA, filename), 'wb') as f:
            np.save(f, wind, allow_pickle = True)
        f.close()
        
        print('***FILES IMPORTED SUCCESS***')
        print('***FILES RESAVE COMPACT SUCCESS***')
    #end
    
    print(f'Longitude range [ {lon_min:.4f} : {lon_max:.4f} ] — shape : {lon.shape}')
    print(f'Latitude  range [ {lat_min:.4f} : {lat_max:.4f} ] — shape : {lat.shape}')
    print(f'Time range [ {times.min()} : {times.max()} ] — shape : {times.shape}')
    print(f'Wind array shape : {wind.shape}')

    sizew = size_obj(wind, unit = 'KiB')
    print(f'Velocity field memory space : {sizew:.4f} KiB')
    sizew = size_obj(wind, unit = 'MiB')
    print(f'Velocity field memory space : {sizew:.4f} MiB')
#end

# Mask land and sea
mask = globe.is_land(lat, lon)
mask_land = np.float32(mask)
mask_sea = np.float32(~mask)

if CROP_IMG is not None:
    
    wind = wind[:, :CROP_IMG, -CROP_IMG:]
    lat, lon = lat[:CROP_IMG, -CROP_IMG:], lon[:CROP_IMG, -CROP_IMG:]
    mask_land = mask_land[-CROP_IMG:, -CROP_IMG:]
    mask_sea = mask_sea[-CROP_IMG:, -CROP_IMG:]
    
    print('Cropped dimensions : {}'.format(wind.shape[-2:]))
#end

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

print('Extent')
print(f'Width  [km] = {width:.2f}')
print(f'Height [km] = {height:.2f}')
print('Resolution')
print(f'Width  : (computed) [km] {width/lat.shape[1]:.2f} ; (average) [km] {diff_km_lat.mean():.2f} ± {diff_km_lat.std():.2f}')
print(f'Width  : (average)  [°]  {diff_deg_lat.mean():.2f} ± {diff_deg_lat.std():.2f}')
print(f'Height : (computed) [km] {height/lon.shape[0]:.2f} ; (average) [km] {diff_km_lon.mean():.2f} ± {diff_km_lon.std():.2f}')
print(f'Heigth : (average)  [°]  {diff_deg_lon.mean():.2f} ± {diff_deg_lon.std():.2f}')

save_netCDF4_dataset(lat, lon, time_range, mask_sea, wind, indices, 'wds', day_start, month_start, year_start)
