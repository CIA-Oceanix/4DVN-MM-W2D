
import os
import numpy as np
import pickle
import torch

gvs          = pickle.load(open(os.path.join(os.getcwd(), 'CONSTANTS', 'CONSTANTS.pkl'), 'rb'))
FORMAT_SIZE  = gvs['FORMAT_SIZE']
LATENT_SPACE = gvs['LATENT_SPACE']
TIME_TAG     = gvs['TIME_TAG']



def daily_format(data, verse):
    
    if verse == 'forward':
        
        batch_size = data.shape[0]
        data_size   = data.shape[-1]
        num_columns = data.shape[-1] * FORMAT_SIZE
        data_reformat = torch.zeros((batch_size, num_columns))
        
        for i in range(batch_size):
            for t in range(FORMAT_SIZE):
                
                begin_idx = np.int32((t * data_size))
                end_idx   = np.int32(data_size + (t * data_size))
                data_reformat[i, begin_idx : end_idx ] = data[i,t,:]
            #end
        #end
        
        return data_reformat
    
    if verse == 'backward':
        
        batch_size = data.shape[0]
        data_size = np.int32( data.shape[-1] / FORMAT_SIZE )
        data_reformat = torch.zeros((batch_size, FORMAT_SIZE, data_size))
        
        for i in range(batch_size):
            for t in range(FORMAT_SIZE):
                
                begin_idx = np.int32(t * data_size)
                end_idx   = np.int32(data_size + (t * data_size))
                data_reformat[i,t,:] = data[i, begin_idx : end_idx ]
            #end
        #end
        
        return data_reformat
    #end
#end


def get_mask_TI(mask_x, mask_y, mask_u, t):
    
    mask_xt = mask_x[:,t,:]; mask_xt_cmpt = mask_xt[:,0].reshape(-1,1)
    mask_yt = mask_y[:,t,:]; mask_yt_cmpt = mask_yt[:,0].reshape(-1,1)
    mask_ut = mask_u[:,t,:]; mask_ut_cmpt = torch.logical_or(mask_xt_cmpt, mask_yt_cmpt).type(torch.float32)
    
    return mask_xt, mask_yt, mask_ut, mask_xt_cmpt, mask_yt_cmpt, mask_ut_cmpt
#end

def get_mask_TD(mask_x, mask_y, mask_u, Nx, Ny):
    
    mask_x_cmpt = mask_x[:, list(np.arange(FORMAT_SIZE) * Nx) ]
    mask_y_cmpt = mask_y[:, list(np.arange(FORMAT_SIZE) * Ny) ]
    mask_u_cmpt = torch.logical_or(mask_x_cmpt, mask_y_cmpt).type(torch.float32)
    
    return mask_x_cmpt, mask_y_cmpt, mask_u_cmpt
#end


def get_latents(zp_x, zp_y, mask_x, mask_y):
    
    if torch.all(mask_x == 0) and not torch.all(mask_y == 0):
        zp_x_input = torch.zeros_like(zp_x)
        zp_y_input = zp_y
    if not torch.all(mask_x == 0) and torch.all(mask_y == 0):
        zp_x_input = zp_x
        zp_y_input = torch.zeros_like(zp_y)
    if not torch.all(mask_x == 0) and not torch.all(mask_y == 0):
        zp_x_input = zp_x
        zp_y_input = zp_y
    if torch.all(mask_x == 0) and torch.all(mask_y == 0):
        zp_x_input = torch.zeros_like(zp_x)
        zp_y_input = torch.zeros_like(zp_y)
    #end
    
    return zp_x_input, zp_y_input
#end

