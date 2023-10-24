import os
import numpy as np
import pickle

import netCDF4 as nc
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    DEVICE  = torch.device('cuda')
else:
    DEVICE  = torch.device('cpu')
#end


class W2DSimuDataset_WindComponents(Dataset):
    
    def __init__(self, data, normalize, case_wmod):
        
        if normalize:
            print('DATA PREPROCESSING: normalize by std')
            wind2D = self.normalize(data, case_wmod)
            self.wind2D = wind2D
        else:
            print('NO DATA PREPROCESSING (normalization)')
            self.wind2D = data
            self.normparams = {}
        #end
        
        self.numitems = self.wind2D.__len__()
        self.to_tensor()
    #end
    
    def __len__(self):
        return self.numitems
    #end
    
    def __getitem__(self, idx):
        return (self.wind2D[idx,:,:,:,0], self.wind2D[idx,:,:,:,1])
    #end
    
    def normalize(self, data, case_wmod):
        
        normparams = dict()
        
        data_std  = np.sqrt(data[:,:,:,:,0]**2 + data[:,:,:,:,1]**2).std()
        data[:,:,:,:,0] = data[:,:,:,:,0] / data_std
        data[:,:,:,:,1] = data[:,:,:,:,1] / data_std
        
        normparams.update({'std'  : data_std})
        
        self.normparams = normparams
        return data
    #end
    
    def to_tensor(self):
        self.wind2D = torch.Tensor(self.wind2D).type(torch.float32).to(DEVICE)
    #end
    
    def get_normparams(self):
        return self.normparams
    #end
#end

class W2DSimuDataModule(pl.LightningDataModule):
    
    def __init__(self, path_data, cparams, timesteps = 24, normalize = True):
        super(W2DSimuDataModule, self).__init__()
        
        self.path_data     = path_data
        self.region_case   = cparams.REGION_CASE
        self.region_extent = cparams.REGION_EXTENT_PX
        self.hr_mask_mode  = cparams.HR_MASK_MODE
        self.batch_size    = cparams.BATCH_SIZE
        self.test_days     = cparams.TEST_DAYS
        self.val_days      = cparams.VAL_DAYS
        self.timesteps     = timesteps
        self.normalize     = normalize
        self.data_name     = cparams.DATASET_NAME_W2D
        self.shapeData     = None
        self.wind_modulus  = cparams.WIND_MODULUS
        self.Dataset_class = W2DSimuDataset_WindComponents
        
        self.setup()
    #end
    
    def get_shapeData(self):
        
        if self.shapeData is not None:
            return self.shapeData
        else:
            raise ValueError('Shape data is None, likely not initialized instance')
        #end
    #ends
    
    def get_mask_land(self):
        
        return self.mask_land
    #end
    
    def get_land_and_buoy_positions(self):
        
        return self.mask_land, self.buoy_positions
    #end
    
    def setup(self, stage = None):
        
        print('\nImporting dataset ...')
        
        # NetCDF4 dataset
        ds_wind2D = nc.Dataset(os.path.join(self.path_data, 'winds_24h', self.data_name), 'r')
        wind2D = np.array(ds_wind2D['wind'])
        mask_land = np.array(ds_wind2D['mask_land'])
        region_lat = np.array(ds_wind2D['lat'])
        region_lon = np.array(ds_wind2D['lon'])
        ds_wind2D.close()
        
        if self.region_case == 'coast-MA':
            wind2D = wind2D[:,-self.region_extent:, -self.region_extent:,:]
            mask_land = mask_land[-self.region_extent:, -self.region_extent:]
            region_lat = region_lat[-self.region_extent:, -self.region_extent:]
            region_lon = region_lon[-self.region_extent:, -self.region_extent:]
        else:
            raise ValueError('Not implemented yet')
        #end
        
        print('Dataset imported. Extracting train/test/val sets ...')
        shape = wind2D.shape[1:3]
        self.shapeData = (self.batch_size, self.timesteps, *tuple(shape))
        
        print(wind2D.shape)
        
        n_test  = np.int32(24 * self.test_days)
        n_train = np.int32(wind2D.__len__() - n_test)
        n_val   = np.int32(24 * self.val_days)
        n_train = np.int32(n_train - n_val)
        
        train_set = wind2D[:n_train, :,:,:]
        val_set   = wind2D[n_train : n_train + n_val, :,:,:]
        test_set  = wind2D[n_train + n_val : n_train + n_val + n_test, :,:,:]
        
        print(train_set.shape, test_set.shape, val_set.shape)
        
        train_set = self.extract_time_series(train_set, 36, n_train // 24)
        val_set   = self.extract_time_series(val_set, 36, n_val // 24)
        test_set  = self.extract_time_series(test_set, 36, self.test_days)
        
        print('Train dataset shape : ', train_set.shape)
        print('Val   dataset shape : ', val_set.shape)
        print('Test  dataset shape : ', test_set.shape)
        print('\nInstantiating torch Datasets ...')
        
        self.mask_land = mask_land
        self.buoy_positions = self.get_buoy_locations(region_lat, region_lon)
        self.train_dataset  = self.Dataset_class(train_set, self.normalize, self.wind_modulus)
        self.val_dataset    = self.Dataset_class(val_set,   self.normalize, self.wind_modulus)
        self.test_dataset   = self.Dataset_class(test_set,  self.normalize, self.wind_modulus)
        self.save_nparams()
    #end
    
    def denormalize(self, data):
        
        normparams = self.test_dataset.get_normparams()
        return data * normparams['std']
    #end
    
    def extract_time_series(self, wind_data, ts_length, num_subseries, random_extract = False):
        
        if random_extract:
            new_wind = np.zeros((num_subseries, ts_length, *wind_data.shape[-3:]))
            for i in range(num_subseries):
                idx_series_start = np.random.randint(0, wind_data.shape[0] - ts_length)
                new_series = wind_data[idx_series_start : idx_series_start + ts_length, :,:,:]
                new_wind[i,:,:,:] = new_series
            #end
        else:
            new_wind = np.zeros((num_subseries - 2, ts_length, *wind_data.shape[-3:]))
            for t in range(num_subseries - 2):
                t_true = 18 + 24 * t
                new_wind[t,:,:,:] = wind_data[t_true : t_true + ts_length,:,:,:]
            #end
        #end
        
        return new_wind
    #end
    
    def get_buoy_locations(self, lat, lon):
        
        def coord_to_index(lat, lon, bcoords):
            
            lat_coord = np.zeros(lat.shape)
            lon_coord = np.zeros(lon.shape)
            lat_coord[(lat > bcoords[0] - 0.03) & (lat < bcoords[0] + 0.03)] = 1.
            lon_coord[(lon > bcoords[1] - 0.03) & (lon < bcoords[1] + 0.03)] = 1.
            coord = np.float32(np.bool_(lat_coord) & np.bool_(lon_coord))
            
            if coord.max() == 0.0:
                return None
            #end
            
            index = np.unravel_index(coord.argmax(), coord.shape)
            return index
        #end
        
        dir_buoys = os.path.join(self.path_data, 'wind_buoys')
        
        if self.hr_mask_mode.__class__ is list:
            buoyID = str(self.hr_mask_mode[1])
            
            coords = np.genfromtxt(os.path.join(dir_buoys, buoyID, 'coords.txt'), delimiter = ',')
            buoy_positions = coord_to_index(lat, lon, coords)
            buoy_positions = np.array(buoy_positions).reshape(1,2)
            
        elif self.hr_mask_mode == 'buoys' or self.hr_mask_mode == 'zeroes':
            
            lcoords = list()
            for cdir in os.listdir(dir_buoys):
                coords = np.genfromtxt(os.path.join(dir_buoys, cdir, 'coords.txt'), delimiter = ',')
                buoy_position = coord_to_index(lat, lon, coords)
                
                if buoy_position is not None:
                    lcoords.append(buoy_position)
                #end                
            #end
            
            buoy_positions = np.array(lcoords)
            logical_flag = np.ones(buoy_positions.shape[0]).reshape(-1,1)
            buoy_positions = np.concatenate((buoy_positions, logical_flag), axis = 1).astype(np.int32)
            
        else:
            return None
        #end
        
        return buoy_positions
    #end
    
    def save_nparams(self):
        
        if self.wind_modulus:
            addendum = '-wm'
        else:
            addendum = '-uv'
        #end
        
        with open(os.path.join(self.path_data, 'winds_24h', f'normparams-test{addendum}.pkl'), 'wb') as f:
            pickle.dump(self.test_dataset.get_normparams(), f)
        f.close()
    #end
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, generator = torch.Generator(DEVICE))
    #end
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size = self.batch_size, generator = torch.Generator(DEVICE))
    #end
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size = self.batch_size, generator = torch.Generator(DEVICE))
    #end
#end



#------------------------------------------------------------------------------
# TIP
# CASE HISTOGRAMS
#------------------------------------------------------------------------------

class WPDFSimuData(Dataset):
    
    def __init__(self, data_hist, data_field, normalize):
        
        self.data_hist  = data_hist
        
        if normalize:
            print('DATA PREPROCESSING: divide by (modulus) std')
            data_field = self.normalize(data_field)
            self.data_field = data_field
        else:
            print('NO DATA PREPROCESSING (normalization)')
            self.data_field = data_field
            self.normparams = {'std' : 1.}
        #end
        
        self.nitems = self.data_hist.shape[0]
        self.to_tensor()
    #end
    
    def __len__(self):
        return self.nitems
    #end
    
    def __getitem__(self, idx):
        return self.data_hist[idx,:,:,:,:], (self.data_field[idx,:,:,:,0], self.data_field[idx,:,:,:,1])
    #end
    
    def normalize(self, data):
        
        normparams = dict()
        
        data_std  = np.sqrt(data[:,:,:,:,0]**2 + data[:,:,:,:,1]**2).std()
        data[:,:,:,:,0] = data[:,:,:,:,0] / data_std
        data[:,:,:,:,1] = data[:,:,:,:,1] / data_std
        normparams.update({'std' : data_std})
        
        self.normparams = normparams
        return data
    #end
    
    def to_tensor(self):
        self.data_hist  = torch.Tensor(self.data_hist).type(torch.float32).to(DEVICE)
        self.data_field = torch.Tensor(self.data_field).type(torch.float32).to(DEVICE)
    #end
    
    def get_normparams(self):
        return self.normparams
    #end
#end


class WPDFSimuDataModule(pl.LightningDataModule):
    
    def __init__(self, path_data, cparams, timesteps = 24, normalize = False):
        super(WPDFSimuDataModule, self).__init__()
        
        self.path_data     = path_data
        self.region_case   = cparams.REGION_CASE
        self.region_extent = cparams.REGION_EXTENT_PX
        self.lr_kernelsize = cparams.LR_KERNELSIZE
        self.hr_mask_mode  = cparams.HR_MASK_MODE
        self.batch_size    = cparams.BATCH_SIZE
        self.test_days     = cparams.TEST_DAYS
        self.val_days      = cparams.VAL_DAYS
        self.timesteps     = timesteps
        self.data_name     = cparams.DATASET_NAME_PDF
        self.shapeData     = None
        self.normalize     = normalize
        self.wind_modulus  = cparams.WIND_MODULUS
        self.Dataset_class = WPDFSimuData
        
        self.setup()
    #end
    
    def get_shapeData(self):
        
        if self.shapeData is not None:
            return self.shapeData
        else:
            raise ValueError('Shape data is None, likely not initialized instance')
        #end
    #ends
    
    def setup(self):
        
        ds = nc.Dataset(os.path.join(self.path_data, 'winds_24h', self.data_name))
        wind_hist = np.array(ds['hist_wind_hr'])
        wind_hr   = np.array(ds['hr_wind_modulus'])
        mask      = np.array(ds['mask_land'])
        
        timesteps, height_lr, width_lr, bins = wind_hist.shape
        self.shapeData = (self.batch_size, self.timesteps, height_lr, width_lr, bins)
        
        n_test  = np.int32(24 * self.test_days)
        n_train = np.int32(wind_hist.__len__() - n_test)
        n_val   = np.int32(24 * self.val_days)
        n_train = np.int32(n_train - n_val)
        
        hwind_train_set = wind_hist[:n_train, :,:,:]
        hwind_val_set   = wind_hist[n_train : n_train + n_val, :,:,:]
        hwind_test_set  = wind_hist[n_train + n_val : n_train + n_val + n_test, :,:,:]
        
        mwind_train_set = wind_hr[:n_train, :,:,:]
        mwind_val_set   = wind_hr[n_train : n_train + n_val, :,:,:]
        mwind_test_set  = wind_hr[n_train + n_val : n_train + n_val + n_test, :,:,:]
        
        hwind_train_set = self.extract_time_series(hwind_train_set, 36, n_train // 24,  'hist')
        hwind_val_set   = self.extract_time_series(hwind_val_set,   36, n_val // 24,    'hist')
        hwind_test_set  = self.extract_time_series(hwind_test_set,  36, self.test_days, 'hist')
        
        mwind_train_set = self.extract_time_series(mwind_train_set, 36, n_train // 24,  'field')
        mwind_val_set   = self.extract_time_series(mwind_val_set,   36, n_val // 24,    'field')
        mwind_test_set  = self.extract_time_series(mwind_test_set,  36, self.test_days, 'field')
        
        print('Train dataset shape : ', hwind_train_set.shape, mwind_train_set.shape)
        print('Val   dataset shape : ', hwind_val_set.shape,   mwind_val_set.shape)
        print('Test  dataset shape : ', hwind_test_set.shape,  mwind_test_set.shape)
        print('\nInstantiating torch Datasets ...')
        
        self.mask_land       = mask
        self.buoys_positions = None
        self.train_dataset   = self.Dataset_class(hwind_train_set, mwind_train_set, self.normalize)
        self.test_dataset    = self.Dataset_class(hwind_test_set,  mwind_test_set, self.normalize)
        self.val_dataset     = self.Dataset_class(hwind_val_set,   mwind_val_set, self.normalize)
        self.save_nparams()        
    #end
    
    def extract_time_series(self, wind_data, ts_length, num_subseries, mod, random_extract = False):
        
        if random_extract:
            new_wind = np.zeros((num_subseries, ts_length, *wind_data.shape[-3:]))
            for i in range(num_subseries):
                idx_series_start = np.random.randint(0, wind_data.shape[0] - ts_length)
                new_series = wind_data[idx_series_start : idx_series_start + ts_length, :,:,:]
                new_wind[i,:,:,:] = new_series
            #end
        else:
            if mod == 'hist':
                new_wind = np.zeros((num_subseries - 2, ts_length, *wind_data.shape[-3:]))
            elif mod == 'field':
                new_wind = np.zeros((num_subseries - 2, ts_length, *wind_data.shape[-3:]))
            #end
            
            for t in range(num_subseries - 2):
                t_true = 18 + 24 * t
                new_wind[t] = wind_data[t_true : t_true + ts_length]
            #end
        #end
        
        return new_wind
    #end
    
    def save_nparams(self):
        
        with open(os.path.join(self.path_data, 'winds_24h', 'normparams-test-hmod.pkl'), 'wb') as f:
            pickle.dump(self.test_dataset.get_normparams(), f)
        f.close()
    #end
    
    def get_normparams(self, stage):
        if stage == 'train':
            return self.train_dataset.get_normparams()
        if stage == 'test':
            return self.test_dataset.get_normparams()
        #end
    #end
    
    def get_land_and_buoy_positions(self):
        return self.mask_land, self.buoys_positions
    #end
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size = self.batch_size, generator = torch.Generator(DEVICE))
    #end
    
    def val_dataloader(self):
        
        return DataLoader(self.val_dataset, batch_size = self.batch_size, generator = torch.Generator(DEVICE))
    #end
    
    def test_dataloader(self):
        
        return DataLoader(self.test_dataset, batch_size = self.batch_size, generator = torch.Generator(DEVICE))
    #end
#end
        