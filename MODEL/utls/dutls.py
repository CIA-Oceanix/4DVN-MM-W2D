import os
import numpy as np
import pickle

import netCDF4 as nc
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    DEVICE  = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    WORKERS = 32
else:
    DEVICE  = torch.device('cpu')
    WORKERS = 8
#end


class W2DSimuDataset_WindModulus(Dataset):
    
    def __init__(self, data, normalize):
        
        # normalize
        wind2D = self.normalize(data)
        self.wind2D = np.array(wind2D, dtype = np.float32)
        
        self.numitems = np.int32(self.wind2D.__len__())
        # self.to_tensor()
    #end
    
    def __len__(self):
        return self.numitems
    #end
    
    def __getitem__(self, idx):
        return self.wind2D[idx]
    #end
    
    def normalize(self, data):
        
        normparams = dict()
        
        # data_min = data.min(); normparams.update({'min' : data_min})
        # data_max = data.max(); normparams.update({'max' : data_max})
        
        # data = (data - data_min) / (data_max - data_min)
        
        data_std  = data.std()
        data = data / data_std
        
        normparams.update({'std' : data_std})
        
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


class W2DSimuDataset_WindComponents(Dataset):
    
    def __init__(self, data, normalize, case_wmod):
        
        wind2D = self.normalize(data, case_wmod)
        self.wind2D = wind2D
        
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
        
        if not case_wmod:
            # Dirty trick to positivize the components distribution
            # Rather sure it's illigal
            # print('Dirty positivizing trick')
            # data[:,:,:,:,0] += np.abs(data[:,:,:,:,0].min())
            # data[:,:,:,:,1] += np.abs(data[:,:,:,:,1].min())
            pass
        #end
        
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
    
    def __init__(self, path_data, cparams, timesteps = 24, normalize = False):
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
        self.data_name     = cparams.DATASET_NAME
        self.shapeData     = None
        self.wind_modulus  = cparams.WIND_MODULUS
        # if cparams.WIND_MODULUS:
        #     self.Dataset_class = W2DSimuDataset_WindModulus
        # else:
        #     self.Dataset_class = W2DSimuDataset_WindComponents
        # #end
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
        
        n_test  = np.int32(24 * self.test_days)
        n_train = np.int32(wind2D.__len__() - n_test)
        n_val   = np.int32(24 * self.val_days)
        n_train = np.int32(n_train - n_val)
        
        train_set = wind2D[:n_train, :,:,:]
        val_set   = wind2D[n_train : n_train + n_val, :,:,:]
        test_set  = wind2D[n_train + n_val : n_train + n_val + n_test, :,:,:]
        
        train_set = self.extract_time_series(train_set, 36, n_train // 24)
        val_set   = self.extract_time_series(val_set, 36, n_val // 24)
        test_set  = self.extract_time_series(test_set, 36, self.test_days)
        
        # if self.wind_modulus:
        #     train_set = np.sqrt(train_set[:,:,:,:,0]**2 + train_set[:,:,:,:,1]**2)
        #     val_set   = np.sqrt(val_set[:,:,:,:,0]**2 + val_set[:,:,:,:,1]**2)
        #     test_set  = np.sqrt(test_set[:,:,:,:,0]**2 + test_set[:,:,:,:,1]**2)
        # #end
        
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
            
        elif self.hr_mask_mode == 'buoys' or self.hr_mask_mode == 'buoysMM':
            
            lcoords = list()
            for cdir in os.listdir(dir_buoys):
                coords = np.genfromtxt(os.path.join(dir_buoys, cdir, 'coords.txt'), delimiter = ',')
                buoy_position = coord_to_index(lat, lon, coords)
                
                if buoy_position is not None:
                    lcoords.append(buoy_position)
                #end
                
            #end
            buoy_positions = np.array(lcoords)
            
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


