import os
import numpy as np
import pickle

import netCDF4 as nc
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    DEVICE  = torch.device('cuda')
    WORKERS = 0
else:
    DEVICE  = torch.device('cpu')
    WORKERS = 8
#end



class W2DSimuDataset(Dataset):
    
    def __init__(self, data, normalize):
        
        # normalization parameters, used to normalize and denormalize data
        self.pparams = dict()
        
        # normalize
        wind2D = self.normalize_imgwise(data, 'wind_2D_hr')
        self.wind2D = wind2D
        
        self.numitems = self.wind2D.__len__()
        self.to_tensor()
    #end
    
    def __len__(self):
        
        return self.numitems
    #end
    
    def __getitem__(self, idx):
        
        return self.wind2D[idx]
    #end
    
    def normalize_imgwise(self, data, name):
        
        normparams = {
            'min'  : np.zeros(data.shape[0]),
            'max'  : np.zeros(data.shape[0]),
            'mean' : 0.,
            'std'  : 0.
        }
        
        for i in range(data.shape[0]):
            normparams['min'][i] = data[i].min()
            normparams['max'][i] = data[i].max()
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
        #end
        
        img_mean = data.mean()
        img_std  = data.std()
        normparams['mean'] = img_mean
        normparams['std']  = img_std
        data = (data - img_mean) / img_std
        
        self.normparams = normparams
        return data
    #end
    
    def normalize(self, data, name):
        
        dmax = data.max()
        dmin = data.min()
        self.pparams.update({name : {'max' : dmax, 'min' : dmin}})
        
        return (data - dmin) / (dmax - dmin)
    #end
    
    def denormalize(self, data, name):
        
        dmin = self.pparams[name]['min']
        dmax = self.pparams[name]['max']
        
        return dmin + (dmax - dmin) * data
    #end
    
    def to_tensor(self):
        
        self.wind2D = torch.Tensor(self.wind2D).type(torch.float32).to(DEVICE)
    #end
    
    def get_normparams(self):
        return self.normparams
    #end
#end


class W2DSimuDataModule(pl.LightningDataModule):
    
    def __init__(self, path_data, cparams, normalize = False):
        super(W2DSimuDataModule, self).__init__()
        
        self.path_data     = path_data
        self.region_case   = cparams.REGION_CASE
        self.region_extent = cparams.REGION_EXTENT_PX
        self.hr_mask_mode  = cparams.HR_MASK_MODE
        self.batch_size    = cparams.BATCH_SIZE
        self.ttsplit       = cparams.TR_TE_SPLIT
        self.tvsplit       = cparams.TR_VA_SPLIT
        self.normalize     = normalize
        self.data_name     = cparams.DATASET_NAME
        self.shapeData     = None
        
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
        
        # NetCDF4 dataset
        ds_wind2D = nc.Dataset(os.path.join(self.path_data, 'winds_24h', self.data_name), 'r')
        wind2D = np.array(ds_wind2D['wind'])
        mask_land = np.array(ds_wind2D['mask_land'])
        region_lat = np.array(ds_wind2D['lat'])
        region_lon = np.array(ds_wind2D['lon'])
        ds_wind2D.close()
        shape = wind2D.shape[-2:]
        
        if self.region_case == 'coast-MA':
            wind2D = wind2D.reshape(-1, 24, *tuple(shape))[:,:, -self.region_extent:, -self.region_extent:]
            mask_land = mask_land[-self.region_extent:, -self.region_extent:]
            region_lat = region_lat[-self.region_extent:, -self.region_extent:]
            region_lon = region_lon[-self.region_extent:, -self.region_extent:]
        else:
            raise ValueError('Not implemented yet')
        #end
        
        self.shapeData = (self.batch_size, 24, *tuple(shape))
        
        n_test  = np.int32(wind2D.__len__() * self.ttsplit)
        n_train = np.int32(wind2D.__len__() - n_test)
        n_val   = np.int32(n_train * self.tvsplit)
        n_train = np.int32(n_train - n_val)
        
        train_set = wind2D[:n_train, :, :]
        val_set   = wind2D[n_train : n_train + n_val, :, :]
        test_set  = wind2D[n_train + n_val : n_train + n_val + n_test, :, :]
        
        print('Train dataset shape : ', train_set.shape)
        print('Val   dataset shape : ', val_set.shape)
        print('Test  dataset shape : ', test_set.shape)
        print()
        
        with open(os.path.join(self.path_data, 'testset.pkl'), 'wb') as f:
            pickle.dump(self.test_dataset, f)
        f.close()
        
        self.mask_land = mask_land
        self.buoy_positions = self.get_buoy_locations(region_lat, region_lon)
        self.train_dataset  = W2DSimuDataset(train_set, normalize = self.normalize)
        self.val_dataset    = W2DSimuDataset(val_set,   normalize = self.normalize)
        self.test_dataset   = W2DSimuDataset(test_set,  normalize = self.normalize)
        self.save_nparams()
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
            
        elif self.hr_mask_mode == 'buoys':
            
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
        
        with open(os.path.join(self.path_data, f'{self.data_name}-normparams-test.pkl'), 'wb') as f:
            pickle.dump(self.test_dataset.get_normparams(), f)
        f.close()
    #end
    
    def train_dataloader(self):
        
        return DataLoader(self.train_dataset, batch_size = self.batch_size)
    #end
    
    def val_dataloader(self):
        
        return DataLoader(self.val_dataset, batch_size = self.batch_size)
    #end
    
    def test_dataloader(self):
        
        return DataLoader(self.test_dataset, batch_size = self.batch_size)
    #end
#end