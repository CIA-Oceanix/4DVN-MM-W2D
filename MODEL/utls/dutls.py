
import os
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    DEVICE  = torch.device('cuda')
    WORKERS = 64
else:
    DEVICE  = torch.device('cpu')
    WORKERS = 8
#end


class W2DSimuDataset(Dataset):
    
    def __init__(self, data, lr_factor, normalize):
        
        # normalization parameters, used to normalize and denormalize data
        self.pparams = dict()
        
        # load data, convert to torch.Tensor and
        # down-sample to obtain low-resoluted spatial fields
        wind_2D_hr = data
        wind_2D_hr = torch.Tensor(wind_2D_hr).type(torch.float32).to(DEVICE)
        # wind_2D_lr = torch.nn.AvgPool2d(lr_factor)(wind_2D_hr)
        
        # normalize
        wind_2D_hr = self.normalize(wind_2D_hr, 'wind_2D_hr')
        # wind_2D_lr = self.normalize(wind_2D_lr, 'wind_2D_lr')
        
        self.wind2D_hr = wind_2D_hr
        # self.wind2D_lr = wind_2D_lr
        
        self.numitems = wind_2D_hr.__len__()
    #end
    
    def __len__(self):
        
        return self.numitems
    #end
    
    def __getitem__(self, idx):
        
        return self.wind2D_hr[idx] #, self.wind2D_lr[idx]
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
#end


class W2DSimuDataModule(pl.LightningDataModule):
    
    def __init__(self, path_data, batch_size, 
                 ttsplit = 0.33, 
                 tvsplit = 0.15,
                 lr_factor = 6, 
                 normalize = True):
        super(W2DSimuDataModule, self).__init__()
        
        self.path_data  = path_data
        self.batch_size = batch_size
        self.ttsplit    = ttsplit
        self.tvsplit    = tvsplit
        self.lr_factor  = lr_factor
        self.normalize  = normalize
        
        self.setup()
    #end
    
    def setup(self):
        
        wind_2D_hr = np.load(open(os.path.join(self.path_data, 
                                  'patch_modwind2D_24h.npy'), 'rb'))
        
        shape = wind_2D_hr.shape[-2:]
        wind_2D_hr = wind_2D_hr.reshape(-1, 24, shape[0], shape[1])
        
        n_test  = np.int32(wind_2D_hr.__len__() * self.ttsplit)
        n_train = np.int32(wind_2D_hr.__len__() - n_test)
        n_val   = np.int32(n_train * self.tvsplit)
        n_train = np.int32(n_train - n_val)
        
        train_set = wind_2D_hr[:n_train, :, :]
        val_set   = wind_2D_hr[n_train : n_train + n_val, :, :]
        test_set  = wind_2D_hr[n_train + n_val : n_train + n_val + n_test, :, :]
                
        self.train_dataset = W2DSimuDataset(train_set,
                                            lr_factor = self.lr_factor, 
                                            normalize = self.normalize)
        self.val_dataset = W2DSimuDataset(val_set,
                                          lr_factor = self.lr_factor, 
                                          normalize = self.normalize)
        self.test_dataset = W2DSimuDataset(test_set,
                                           lr_factor = self.lr_factor, 
                                           normalize = self.normalize)
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
