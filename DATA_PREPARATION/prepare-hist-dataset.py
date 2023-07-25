
import numpy as np
import torch
import torch.nn.functional as F
import netCDF4 as nc
import pickle
import os
import json
from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from mpl_toolkits.axes_grid1 import make_axes_locatable

import argparse
from dotenv import load_dotenv
load_dotenv(os.path.join(os.getcwd(), 'config.env'))
PATH_DATA = os.getenv('PATH_DATA')
PATH_HOME = os.getenv('PATH_HOME')
with open(os.path.join(PATH_HOME, 'cparams.json'), 'r') as f:
    CPARAMS = json.load(f)
f.close()
_cparams = namedtuple('config_params', CPARAMS)
cparams  = _cparams(**CPARAMS)
BINS = cparams.WIND_BINS



def imshow_cb(ax_subplot, img, title, cmap = 'viridis', vmax = None, vmin = None, titlepad = None):
    
    if vmax is None and vmin is not None:
        subplot = ax_subplot.imshow(img, cmap = cmap)
    else:
        subplot = ax_subplot.imshow(img, cmap = cmap, vmax = vmax, vmin = vmin)
    #end
    
    divider = make_axes_locatable(ax_subplot)
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    cb = fig.colorbar(subplot, cax = cax, orientation = 'vertical')
    for t in cb.ax.get_yticklabels():
        t.set_fontsize(12)
    #end
    
    ax_subplot.set_xticks([]); ax_subplot.set_yticks([])
    
    if titlepad is None:
        ax_subplot.set_title(title, fontsize = 12)
    else:
        ax_subplot.set_title(title, pad = titlepad)
    #end
    
    return ax_subplot
#end

def get_histogram(data, bins, to_beaufort_scale = False):
    
    histogram = torch.zeros(bins.shape[0] - 1)
    for eidx in range(bins.__len__() - 1):
        histogram[eidx] = torch.numel(data[(data > bins[eidx]) & (data <= bins[eidx + 1])])
    #end
    
    return histogram
#end

def make_hist(data_, bins, normalized = True):
    
    if bins.__class__ is not torch.Tensor:
        bins = torch.Tensor(bins)
    #end
    
    h = get_histogram(data_, bins = bins)
    
    if normalized:
        return h.div(h.sum())
    else:
        return h
    #end
#end

def fieldsHR2hist(data_field, kernel_size, bins, progbars = False):
    '''
    Takes as input tensors of dimension
        (batch_size, timesteps, height, width)
    '''
    
    def lr_dim(dim, ks_):
        return np.int32(np.floor( (dim - ks_) / ks_ + 1 ))
    #end
    
    timesteps, heigth, width = data_field.shape
    height_lr, width_lr = lr_dim(heigth, kernel_size), lr_dim(width, kernel_size)
    data_hist_hr = torch.zeros((timesteps, height_lr, width_lr, bins.__len__() - 1))
    data_hist_lr = torch.zeros((timesteps, height_lr, width_lr, bins.__len__() - 1))
    
    # loop to prepare histogram data
    progbar_timesteps = tqdm(range(timesteps),  desc = 'Timesteps ', position = 0, leave = True)
    
    for t in progbar_timesteps:
        
        i_start = 0
        for i in range(height_lr):
            
            j_start = 0
            for j in range(width_lr):
                
                i_end = i_start + kernel_size
                j_end = j_start + kernel_size
                
                try:
                    this_wind_pixel = data_field[t, i_start:i_end ,j_start:j_end]
                except:
                    this_wind_pixel = data_field[t, i_start:, j_start:]
                #end
                
                hist_hr = make_hist(this_wind_pixel, bins)
                data_hist_hr[t,i,j,:] = hist_hr
                
                hist_lr = make_hist(this_wind_pixel.mean(), bins)
                data_hist_lr[t,i,j,:] = hist_lr
                
                j_start = j_end
            #end
            
            i_start = i_end
        #end
    #end
    
    return data_hist_hr, data_hist_lr
#end


def save_netCDF4_dataset(lat, lon, time, mask, w_hist_hr, w_hist_lr, w_lr, indices, ds_name, 
			  day_start, month_start, year_start,
			  day_end, month_end, year_end):
   
    filename = f'{ds_name}_{day_start}-{month_start}-{year_start}_{day_end}-{month_end}-{year_end}.nc'
    if os.path.exists(os.path.join(PATH_DATA, filename)):
        os.remove(os.path.join(PATH_DATA, filename))
        print('Old Dataset removed ...')
    #end
    
    print('Creating new netCDF4 Dataset ...')
    nc_dataset = nc.Dataset(os.path.join(PATH_DATA, f'{filename}'), mode = 'w', format = 'NETCDF4_CLASSIC')
    nc_dataset.createDimension('south-north', lat.shape[0])
    nc_dataset.createDimension('west-east', lon.shape[1])
    nc_dataset.createDimension('time', time.__len__())
    nc_dataset.createDimension('hbins', w_hist_hr.shape[-1])
    
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
    nc_time.units = f'hours_since_{day_start}-{month_start}-{year_start}_to{day_end}-{month_end}-{year_end}_dd-mm-yyyy'
    nc_time.long_name = 'hours'
    nc_hist_wind_hr = nc_dataset.createVariable('hist_wind_hr', np.float32, ('time', 'south-north', 'west-east', 'hbins'))
    nc_hist_wind_hr.units = 'norm_frequencies'
    nc_hist_wind_hr.long_name = 'model_wind_probabilities_hr'
    nc_hist_wind_lr = nc_dataset.createVariable('hist_wind_lr', np.float32, ('time', 'south-north', 'west-east', 'hbins'))
    nc_hist_wind_lr.units = 'norm_frequencies'
    nc_hist_wind_lr.long_name = 'model_wind_probabilities_lr'
    nc_avg_wind = nc_dataset.createVariable('avg_wind', np.float32, ('time', 'south-north', 'west-east'))
    nc_windices = nc_dataset.createVariable('indices', np.int32, ('time',))
    nc_windices.units = 'none'
    nc_windices.long_name = 'indices_of_wind_images'
    
    nc_lat[:,:] = lat
    nc_lon[:,:] = lon
    nc_time[:] = time
    nc_mask[:,:] = mask
    nc_hist_wind_hr[:,:,:,:] = w_hist_hr
    nc_hist_wind_lr[:,:,:,:] = w_hist_lr
    nc_avg_wind[:,:,:] = w_lr
    nc_windices = indices
    
    print('Dataset save. Closing ...')
    nc_dataset.close()
    print('Dataset close.')
#end

def hist_mean_computation(hist, bins):
    
    return (bins * hist).sum()
#end

def get_dataset_days_extrema(ds_name):
    
    day_start   = ds_name[7:9]
    month_start = ds_name[10:12]
    year_start  = ds_name[13:17]
    day_end     = ds_name[18:20]
    month_end   = ds_name[21:23]
    year_end    = ds_name[24:28]
    
    return day_start, month_start, year_start, day_end, month_end, year_end
#end


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-ks', type = int)
    parser.add_argument('-re', type = int, default = 200)
    args = parser.parse_args()
    lr_dsfactor   = args.ks
    region_extent = args.re
    
    if (lr_dsfactor != 10 and lr_dsfactor != 29) or lr_dsfactor is None:
        # raise ValueError('Please provide a valid kernelsize: 10 (LR resolution 30 km) or 29 (LR resolution 100 km)')
        lr_dsfactor = 10
    #end
    
    print()
    print('##################################')
    print('WARNING! Set properly cparams.json')
    print('Kernel size (LR) : {}'.format(lr_dsfactor))
    print('##################################')
    
    dataset_name = cparams.DATASET_NAME
    # dataset_name = 'wds_uv_01-01-2021_01-03-2021.nc'
    ds = nc.Dataset(os.path.join(PATH_DATA, f'{dataset_name}'))
    day_start, month_start, year_start, day_end, month_end, year_end = get_dataset_days_extrema(dataset_name)
    
    lr_dim = np.floor( (region_extent - lr_dsfactor) / lr_dsfactor + 1 )
    reso = np.int32( 3 * region_extent / lr_dim )
    
    w_hr = np.array(ds['wind'])
    lat  = np.array(ds['lat'])
    lon  = np.array(ds['lon'])
    time = np.array(ds['time'])
    idx  = np.array(ds['indices'])
    mask = np.array(ds['mask_land'])
    
    w_hr = np.sqrt(w_hr[:,:,:,0]**2 + w_hr[:,:,:,1]**2)
    w_hr = torch.Tensor(w_hr)[:, -region_extent:, -region_extent:]
    lat  = lat[-region_extent:, -region_extent:]
    lon  = lon[-region_extent:, -region_extent:]
    mask = mask[-region_extent:, -region_extent:]
    
    bins = torch.Tensor([0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32., 35.])
    # bins = torch.Tensor([0., 3., 6.5, 10., 13.5, 16.5, 20., 25., 30., 35.])
    
    # Downsample HR > LR
    w_lr = F.avg_pool2d(w_hr.reshape(1, *tuple(w_hr.shape)), kernel_size = lr_dsfactor).squeeze(0)
    timesteps, height_lr, width_lr = w_lr.shape
    lat_lr = F.avg_pool2d(torch.Tensor(lat).reshape(1, 1, *tuple(lat.shape)), kernel_size = lr_dsfactor).squeeze(0).squeeze(0)
    lon_lr = F.avg_pool2d(torch.Tensor(lat).reshape(1, 1, *tuple(lon.shape)), kernel_size = lr_dsfactor).squeeze(0).squeeze(0)
    mask_lr = F.avg_pool2d(torch.Tensor(mask).reshape(1, 1, *tuple(mask.shape)), kernel_size = lr_dsfactor).squeeze(0).squeeze(0)
    mask_lr[mask_lr <= 0.5] = 0
    mask_lr[mask_lr > 0.5]  = 1
    
    # Histogrammize
    w_hist_hr, w_hist_lr = fieldsHR2hist(w_hr, lr_dsfactor, bins, progbars = True)
    
    # Save dataset
    save_netCDF4_dataset(lat_lr, lon_lr, time, mask_lr, w_hist_hr, w_hist_lr, w_lr, idx, 
                         f'whist-{reso}km', day_start, month_start, year_start, day_end, month_end, year_end)
   
    
   # PLOTS
    xbins = (bins[1:] + bins[:-1]) / 2
    means_hist_computed = torch.zeros(w_lr.shape)
    errors = torch.zeros(w_lr.shape)
    print(w_hist_hr.shape)
    for t in tqdm(range(w_hr.shape[0]),  desc = 'Timesteps ', position = 0, leave = True):
        for i in range(height_lr):
            for j in range(width_lr):
                mean = hist_mean_computation(w_hist_hr[t,i,j], xbins)
                if torch.isnan(mean):
                    print(w_hist_hr[t,i,j])
                    raise ValueError('Nans!!!')
                #end
                means_hist_computed[t,i,j] = mean
                errors[t,i,j] = torch.sqrt((mean - w_lr[t,i,j]).pow(2))
            #end
        #end
    #end
    
    fig, ax = plt.subplots(1,2, figsize = (8.5,4))
    ax[0] = imshow_cb(ax[0], errors.mean(0), 'Average RMSE = {:.4f} m/s'.format(errors.mean()))
    ax[1].scatter(means_hist_computed.flatten().numpy(), w_lr.flatten().numpy())
    ax[1].set_xlabel('Means (histograms) [m/s]')
    ax[1].set_ylabel('Means (LR avg pool 2d) [m/s]')
    ax[1].set_xticks(np.linspace(0, w_hr.max(), 5))
    ax[1].set_yticks(np.linspace(0, w_hr.max(), 5))
    ax[1].plot(np.linspace(0, 20, 5), np.linspace(0, 20, 5), c = 'k', ls = '--', lw = 2)
    fig.tight_layout()
    fig.savefig('./plots/avg_error_means_avgpool_and_histmean.png', format = 'png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
    fig, ax = plt.subplots(1,2, figsize = (6,3))
    ax[0] = imshow_cb(ax[0], w_lr[0], 'Mean AvgPool2d [m/s]')
    ax[1] = imshow_cb(ax[1], means_hist_computed[0], 'Mean histogram [m/s]')
    fig.tight_layout()
    fig.savefig('./plots/fields_means_avgpool_and_histmeans.png', format = 'png', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
    
#end


