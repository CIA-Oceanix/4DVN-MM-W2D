
import os
import numpy as np
from tqdm import tqdm

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
        wind2D = self.normalize(data, 'wind_2D_hr')
        self.wind2D = wind2D
        
        self.numitems = wind2D.__len__()
        self.to_tensor()
    #end
    
    def __len__(self):
        
        return self.numitems
    #end
    
    def __getitem__(self, idx):
        
        return self.wind2D[idx]
    #end
    
    def _normalize(self, data, name):
        
        for i in tqdm(range(data.shape[0])):
            
            data[i] = (data[i] - data[i].min()) / (data[i].max() - data[i].min())
        #end
        
        img_mean = data.mean()
        img_std  = data.std()
        
        data = (data - img_mean) / img_std
        
        print('\nImages normalized')
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
#end


class W2DSimuDataModule(pl.LightningDataModule):
    
    def __init__(self, path_data, cparams, normalize = True):
        super(W2DSimuDataModule, self).__init__()
        
        self.path_data     = path_data
        self.region_case   = cparams.REGION_CASE
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
    
    def setup(self, stage = None):
        
        wind2D = np.load(open(os.path.join(self.path_data, self.data_name), 'rb'))
        mask_land = np.load(open(os.path.join(self.path_data, 'global_mask.npy'), 'rb'))
        shape = wind2D.shape[-2:]
        
        if self.region_case == 'coast':
            wind2D = wind2D.reshape(-1, 24, *tuple(shape))[:,:, -200:, -200:]
            mask_land = mask_land[-200:, -200:]
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
        
        self.mask_land = mask_land
        self.train_dataset = W2DSimuDataset(train_set, normalize = self.normalize)
        self.val_dataset   = W2DSimuDataset(val_set,   normalize = self.normalize)
        self.test_dataset  = W2DSimuDataset(test_set,  normalize = self.normalize)
    #end
    
    def get_mask_land(self):
        return self.mask_land
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
