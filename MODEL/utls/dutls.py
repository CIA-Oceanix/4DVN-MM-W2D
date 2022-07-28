
import os
import pickle
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class W2DSimuDataset(Dataset):
    
    def __init__(self, path_data, lr_factor, normalize):
        
        # normalization parameters, used to normalize and denormalize data
        self.pparams = dict()
        
        # load data, convert to torch.Tensor and
        # down-sample to obtain low-resoluted spatial fields
        wind_2D_hr = np.load(open(os.path.join(path_data, 'patch_modwind2D_24h.npy'), 'rb'))
        wind_2D_hr = torch.Tensor(wind_2D_hr).type(torch.float32).to(DEVICE)
        wind_2D_lr = torch.nn.AvgPool2d(lr_factor)(wind_2D_hr)
        
        # normalize
        wind_2D_hr = self.normalize(wind_2D_hr, 'wind_2D_hr')
        wind_2D_hr = self.normalize(wind_2D_lr, 'wind_2D_lr')
        
        self.wind2D_hr = wind_2D_hr
        self.wind2D_lr = wind_2D_lr
        
        self.numitems = wind_2D_hr.__len__()
    #end
    
    def __len__(self):
        
        return self.numitems
    #end
    
    def __getitem__(self, idx):
        
        return self.wind2D_hr[idx], self.wind2D_lr[idx]
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


class W2DDataModule(pl.LightningDataModule):
    
    def __init__(self, path_data, lr_factor = 6, normalize = True):
        super(W2DDataModule, self).__init__()
        
        wind_2D_dataset = W2DSimuDataset(path_data, lr_factor, normalize)
    #end
#end


class SMData(Dataset):
    
    def __init__(self, path_data, wind_values, data_title,
                 dtype = torch.float32,
                 convert_to_tensor = True,
                 normalize = True):
        
        UPA        = pickle.load( open(os.path.join(path_data, f'UPA_{data_title}.pkl'), 'rb') )
        WIND_situ  = pickle.load( open(os.path.join(path_data, f'WIND_label_SITU_{data_title}.pkl'), 'rb') )
        WIND_ecmwf = pickle.load( open(os.path.join(path_data, f'WIND_label_ECMWF_{data_title}.pkl'), 'rb') )
        
        self.UPA        = np.array( UPA['data'] )
        self.WIND_situ  = np.array( WIND_situ['data'] )
        self.WIND_ecmwf = np.array( WIND_ecmwf['data'] )
        # self.UPA        = UPA['data']
        # self.WIND_situ  = WIND_situ['data']
        # self.WIND_ecmwf = WIND_ecmwf['data']
        
        self.which_wind = WIND_situ['which']
        
        self.preprocess_params = {
                'upa' : UPA['nparms'],
                'wind_situ' : WIND_situ['nparms'],
                'wind_ecmwf' : WIND_ecmwf['nparms']
        }
        
        self.nsamples = self.UPA.__len__()
        self.dtype    = dtype
        
        if convert_to_tensor: self.to_tensor()
    #end
    
    def __len__(self):
        
        return self.nsamples
    #end
    
    def __getitem__(self, idx):
        
        return self.UPA[idx], self.WIND_ecmwf[idx], self.WIND_situ[idx]
    #end
    
    def get_modality_data_size(self, data = None, asdict = False):
        
        N = {
            'upa' : np.int32(self.UPA[0].shape[1]),
            'wind_situ' : np.int32(1),
            'wind_ecmwf' : np.int32(1),
        }
        
        if data is None:
            if asdict:
                return N
            else:
                return N['upa'], N['wind_ecmwf'], N['wind_situ']
            #end
        else:
            return N[data]
        #end
    #end
    
    def to_tensor(self):
        
        # for i in range(self.nsamples):
            
        #     self.UPA[i] = torch.Tensor(self.UPA[i]).type(self.dtype)
        #     self.WIND_situ[i] = torch.Tensor(self.WIND_situ[i]).type(self.dtype)
        #     self.WIND_ecmwf[i] = torch.Tensor(self.WIND_ecmwf[i]).type(self.dtype)
        # #end
        
        self.UPA = torch.Tensor(self.UPA).type(self.dtype).to(device)
        self.WIND_ecmwf = torch.Tensor(self.WIND_ecmwf).type(self.dtype).to(device)
        self.WIND_situ = torch.Tensor(self.WIND_situ).type(self.dtype).to(device)
    #end
    
    def undo_preprocess(self, data_preprocessed, tag):
        
        v_min, v_max = self.preprocess_params[tag]
        data = (v_max - v_min) * data_preprocessed + v_min
        return data
    #end
#end




