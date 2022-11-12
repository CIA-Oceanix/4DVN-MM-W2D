
import sys
import torch
from torch import nn
import numpy as np
from skimage.metrics import structural_similarity


# Crop central patch
def crop_central_patch(img_ts, length):
    
    print('Patch extent : {}'.format(length))
    center_h, center_w = img_ts[0,0].shape[-2] // 2, img_ts[0,0].shape[-1] // 2
    cp_img = img_ts[:,:, center_h - length : center_h + length, center_w - length : center_w + length]
    return cp_img
#end

# B-distances
def get_batched_histograms(img_ts, bins = 30, flat = False):
    
    if not flat:
        img = img_ts.reshape(-1, *img_ts.shape[-2:])
    else:
        img = img_ts.reshape(-1, img_ts.shape[-1])
    #end
    hist = torch.cat([torch.histc(img[i], bins = bins).unsqueeze(0) for i in range(img.shape[0])], dim = 0)
    for i in range(hist.shape[0]):
        hist[i] = hist[i] / hist[i].sum()
    #end
    return hist
#end

def get_obj_size(obj, unit = 'MB'):
    
    if unit == 'KB':
        return sys.getsizeof(obj) / (1e3)
    elif unit == 'MB':
        return sys.getsizeof(obj) / (1e6)
    elif unit == 'GB':
        return sys.getsizeof(obj) / (1e9)
    #end
#end

def relative_gain(perf_base, perf_model, mode = 'lower'):
    
    if mode == 'lower':
        perf_compare = perf_model
        perf_reference = perf_base
    elif mode == 'upper':
        perf_compare = perf_base
        perf_reference = perf_model
    #end
    
    return (1 - perf_compare / perf_reference) * 100.
#end

def Hellinger_distance(h_target, h_output, reduction_dim = 1, mode = 'trineq'):
    
    if reduction_dim is None and h_target.shape.__len__() > 1:
        reduction_dim = 1
    elif reduction_dim is None and h_target.shape.__len__() <= 1:
        reduction_dim = 0
    #end
    
    eps = 1e-5
    # b_coefficient = torch.sum((h_target * h_output).sqrt(), dim = reduction_dim)
    b_coefficient = (torch.sqrt(h_target * h_output)).sum(dim = reduction_dim)
    if torch.any(b_coefficient > 1) or torch.any(b_coefficient < 0):
        raise ValueError('BC can not be > 1 or < 0')
    #end
    
    b_coefficient[b_coefficient < eps] = eps
    
    if mode == 'log':
        b_distance = -1. * torch.log(b_coefficient)
    elif mode == 'trineq':
        b_distance = torch.sqrt(1. - b_coefficient)
    else:
        raise NotImplementedError('Metric selected not valid. Consider setting "log" or "trineq"')
    #end
    
    return b_distance.mean()
#end

def str_sim(target, output):
    
    if target.__class__ is not np.ndarray:
        target = np.array(target)
    #end
    
    if output.__class__ is not np.ndarray:
        output = np.array(output)
    #end
    
    ssims = np.zeros(target.shape[:2])
    for m in range(target.shape[0]):
         for t in range(target.shape[1]):
              ssims[m,t] = structural_similarity(target[m,t], output[m,t])
         #end
    #end
    
    ssim_avg = ssims.mean()
    return ssim_avg
#end

def peak_signal_to_noise_ratio(target, output):
    
    if target.__class__ is not np.ndarray:
        target = np.array(target)
    #end
    
    if output.__class__ is not np.ndarray:
        output = np.array(output)
    #end
    
    psnr = np.zeros(target.shape)
    
    for m in range(target.shape[0]):
        for t in range(target.shape[1]):
            mse = np.mean( np.power(target[m,t] - output[m,t], 2) )
            psnr[m,t] = 10 * np.log10(1. / mse)
        #end
    #end
    
    psnr[psnr > 100] = np.nan
    return np.nanmean(psnr)
#end

def mse(target, output, mask = None, divide_std = True):
    
    mserror = NormLoss()((target - output), mask = mask)
    if divide_std:
        mserror = mserror / target.std()
    #end
    
    return mserror
#end


class _NormLoss(nn.Module):
    
    def __init__(self, dim_item = 2):
        super(_NormLoss, self).__init__()
        
        self.dim_item = dim_item
    #end
    
    def forward(self, item, mask):
        
        if item.__class__ is not torch.Tensor:
            item = torch.Tensor(item)
        #end
        
        if mask.__class__ is not torch.Tensor:
            mask = torch.Tensor(mask)
        #end
        
        if mask is None:
            mask = torch.ones(item.shape)
        #end
        
        # tot_items = mask.shape[2] * mask.shape[3]
        nonzero_items = mask.sum(dim = (2,3))
        nonzero_fract =  1 / nonzero_items #/ tot_items
        
        loss = item.mul(mask).pow(2)
        loss = loss.sum(dim = (2,3))
        loss = loss.mul(nonzero_fract)
        loss = loss.sum(1).mean(0)
        
        return loss
    #end
#end


class NormLoss(nn.Module):
    
    def __init__(self):
        super(NormLoss, self).__init__()
        
    #end
    
    def forward(self, item, mask):
        
        if item.__class__ is not torch.Tensor:
            item = torch.Tensor(item)
        #end
        
        if mask is None:
            mask = torch.ones_like(item)
        elif mask.__class__ is not torch.Tensor:
            mask = torch.Tensor(mask)
        #end
        
        # square
        argument = item.pow(2)
        argument = argument.mul(mask)
        
        if mask.sum() == 0.:
            n_items = 1.
        else:
            n_items = mask.sum().div(24.)
        #end
        
        # sum on:
        #   1. features plane; 
        #   2. timesteps and batches
        # Then mean over effective items
        argument = argument.sum(dim = (2,3))
        argument = argument.sum(dim = (1,0))
        loss = argument.div(n_items)
        
        return loss
    #end
#end