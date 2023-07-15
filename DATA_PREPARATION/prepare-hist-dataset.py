
import numpy as np
import torch
import netCDF4 as nc
import pickle
import os
import json
from tqdm import tqdm
from collections import namedtuple
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from mpl_toolkits.axes_grid1 import make_axes_locatable

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
    
    # h = torch.histogram(data_, bins = bins)
    h = get_histogram(data_, bins = bins)
    h = torch.autograd.Variable(h)
    
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
    data_hist = torch.zeros((timesteps, height_lr, width_lr, bins.__len__() - 1))
    
    # loop to prepare histogram data
    progbar_timesteps = tqdm(range(timesteps),  desc = 'Timesteps   ', position = 0, leave = True)
    
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
                
                hist = make_hist(this_wind_pixel, bins)
                data_hist[t,i,j,:] = hist
                
                j_start = j_end
            #end
            
            i_start = i_end
        #end
    #end
    
    return data_hist
#end

def fieldsHR2avgwinds(data_field, kernel_size):
    
    def lr_dim(dim, ks_):
        return np.int32(np.floor( (dim - ks_) / ks_ + 1 ))
    #end
    
    timesteps, heigth, width = data_field.shape
    height_lr, width_lr = lr_dim(heigth, kernel_size), lr_dim(width, kernel_size)
    data_avg_lr = torch.zeros((timesteps, height_lr, width_lr))
    
    progbar_timesteps = tqdm(range(timesteps),  desc = 'Timesteps   ', position = 0, leave = True)
    
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
                
                data_avg_lr[t,i,j] = torch.mean(this_wind_pixel)
                
                j_start = j_end
            #end
            
            i_start = i_end
        #end
    #end
    
    return data_avg_lr
#end

def save_netCDF4_dataset(lat, lon, time, mask, wind, indices, ds_name, 
			  day_start, month_start, year_start,
			  day_end, month_end, year_end):
   
    filename = f'{ds_name}_{day_start:02d}-{month_start:02d}-{year_start}_{day_end:02d}-{month_end:02d}-{year_end}.nc'
    if os.path.exists(os.path.join(PATH_DATA, filename)):
        os.remove(os.path.join(PATH_DATA, filename))
        print('Old Dataset removed ...')
    #end
    
    print('Creating new netCDF4 Dataset ...')
    nc_dataset = nc.Dataset(os.path.join(PATH_DATA, f'{filename}'), mode = 'w', format = 'NETCDF4_CLASSIC')
    nc_dataset.createDimension('south-north', lat.shape[0])
    nc_dataset.createDimension('west-east', lon.shape[1])
    nc_dataset.createDimension('time', time.__len__())
    nc_dataset.createDimension('wind_components', 2)
    
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
    nc_time.units = f'hours_since_{day_start:02d}-{month_start:02d}-{year_start}_to{day_end:02d}-{month_end:02d}-{year_end:02d}_dd-mm-yyyy'
    nc_time.long_name = 'hours'
    nc_wind = nc_dataset.createVariable('wind', np.float32, ('time', 'south-north', 'west-east', 'wind_components'))
    nc_wind.units = 'm s-1'
    nc_wind.long_name = 'model_wind'
    nc_windices = nc_dataset.createVariable('indices', np.int32, ('time',))
    nc_windices.units = 'none'
    nc_windices.long_name = 'indices_of_wind_images'
    
    nc_lat[:,:] = lat
    nc_lon[:,:] = lon
    nc_time[:] = time
    nc_mask[:,:] = mask
    nc_wind[:,:,:,0] = wind[:,:,:,0]
    nc_wind[:,:,:,1] = wind[:,:,:,1]
    nc_windices = indices
    
    print('Dataset save. Closing ...')
    nc_dataset.close()
    print('Dataset close.')
#end

def hist_mean_computation(hist, bins):
    
    return (bins * hist).sum()
#end


if __name__ == '__main__':
    
    dataset_name = cparams.DATASET_NAME
    ds = nc.Dataset(os.path.join(PATH_DATA, f'{dataset_name}'))
    
    w_hr = np.array(ds['wind'])
    w_hr = np.sqrt(w_hr[:,:,:,0]**2 + w_hr[:,:,:,1]**2)
    w_hr = torch.Tensor(w_hr)[:, -cparams.REGION_EXTENT_PX:, -cparams.REGION_EXTENT_PX:]
    
    bins = torch.Tensor([0., 2., 4., 6., 8., 10., 12., 14., 16., 18., 20., 22., 24., 26., 28., 30., 32.])
    
    w_hist = fieldsHR2hist(w_hr, cparams.LR_KERNELSIZE, bins, progbars = True)
    w_lr   = torch.nn.functional.avg_pool2d(w_hr.reshape(1, *tuple(w_hr.shape)), kernel_size = cparams.LR_KERNELSIZE).squeeze(0)
    timesteps, height_lr, width_lr = w_lr.shape
   
    xbins = (bins[1:] + bins[:-1]) / 2
    means_hist_computed = torch.zeros(w_lr.shape)
    errors = torch.zeros(w_lr.shape)
    print(w_hist.shape)
    for t in range(w_hist.shape[0]):
        for i in range(height_lr):
            for j in range(width_lr):
                mean = hist_mean_computation(w_hist[t,i,j], xbins)
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
    ax[1].set_xticks(np.linspace(0, 20, 5))
    ax[1].set_yticks(np.linspace(0, 20, 5))
    ax[1].plot(np.linspace(0, 20, 5), np.linspace(0, 20, 5), c = 'k', ls = '--', lw = 2)
    fig.tight_layout()
    fig.savefig('./plots/avg_error_means_avgpool_and_histmean.eps', format = 'eps', dpi = 300, bbox_inches = 'tight')
    plt.show()
    
    fig, ax = plt.subplots(1,2, figsize = (6,3))
    ax[0] = imshow_cb(ax[0], w_lr[0], 'Mean AvgPool2d [m/s]')
    ax[1] = imshow_cb(ax[1], means_hist_computed[0], 'Mean histogram [m/s]')
    fig.tight_layout()
    fig.savefig('./plots/fields_means_avgpool_and_histmeans.eps', format = 'eps', dpi = 300, bbox_inches = 'tight')
    plt.show()
#end
    

