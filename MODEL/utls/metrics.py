
import sys
import torch
from torch import nn
import numpy as np
from skimage.metrics import structural_similarity

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
#end


# Crop central patch
def crop_central_patch(img_ts, length):
    
    print('Patch extent : {}'.format(length))
    center_h, center_w = img_ts[0,0].shape[-2] // 2, img_ts[0,0].shape[-1] // 2
    cp_img = img_ts[:,:, center_h - length : center_h + length, center_w - length : center_w + length]
    return cp_img
#end

# B-distances
def get_batched_histograms(img_ts, bins = 30, flat = False):
    
    if img_ts.__class__ is not torch.Tensor:
        img_ts = torch.Tensor(img_ts)
    #end
    
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

def Hellinger_distance_(h_target, h_output, reduction_dim = 1, mode = 'trineq'):
    
    if reduction_dim is None and h_target.shape.__len__() > 1:
        reduction_dim = 1
    elif reduction_dim is None and h_target.shape.__len__() <= 1:
        reduction_dim = 0
    #end
    
    eps = 1e-5
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

class HellingerDistance(nn.Module):
    def __init__(self, n_items = False):
        super(HellingerDistance, self).__init__()
        
        self.n_items = n_items
    #end
    
    def forward(self, target, output, mask = None, mode = 'hd'):
        '''
        Inputs: normalized probabilities, ie histograms summing to 1!!!
        
        Hellinger distance: metric to compare proximity between histograms.
        Given two probability distributions :math:`P` and :math:`Q`, is computed as 
        
            .. math::
                HD(P,Q) = \sqrt{1 - \sum_{x \in X} \sqrt{P(x) Q(x)}}
        
        with :math:`\sum_x P(x) = 1` and :math:`\sum_x Q(x) = 1`.
        '''
        
        if mask is None:
            mask = torch.ones(target.shape)
        #end
        
        if self.n_items:
            nitems = mask.sum()
            if nitems == 0:
                nitems = 1.
            #end
        #end
        
        target[target < 1e-9] = 1e-9
        output[output < 1e-9] = 1e-9
        
        b_coefficient = torch.sqrt( torch.mul(target, output) ).sum(dim = -1)
        if torch.any(b_coefficient > 1.):
            b_coefficient[b_coefficient > 1.] = 1.
        if torch.any(b_coefficient < 0.):
            b_coefficient[b_coefficient < 0.] = 0.
        #end
        
        if mode == 'hd':
            hellinger_distance = torch.sqrt(1. - b_coefficient)
        elif mode == 'bd':
            hellinger_distance = -1. * torch.log(b_coefficient)
        #end
        hellinger_distance = hellinger_distance.unsqueeze(-1).mul(mask)
        
        if self.n_items:
            hd_mean = hellinger_distance.sum().div(nitems)
        else:
            hd_mean = hellinger_distance.mean()
        #end
        
        return hd_mean
    #end
#end

class KLDivLoss(torch.nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    #end
    
    def forward(self, target, output):
        
        if target.__class__ is not torch.Tensor:
            target = torch.Tensor(target)
        if output.__class__ is not torch.Tensor:
            output = torch.Tensor(output)
        #end
        
        target[target < 1e-9] = 1e-9
        output[output < 1e-9] = 1e-9
        kld = target.mul( target.log() - output.log() ).sum(dim = -1)
        kld = kld.mean()
        return kld
    #end
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

def mse(target, output, mask = None, divide_variance = True, divide_nitems = True):
    
    mserror = L2_Loss(divide_nitems = divide_nitems)((target - output), mask = mask)
    if divide_variance:
        mserror = mserror / target.var()
    #end
    
    return mserror
#end

def L1_norm(target, output, mask = None, divide_variance = True, divide_nitems = True):
    
    L1error = L1_Loss(divide_nitems = divide_nitems)((target - output), mask = mask)
    if divide_variance:
        L1error = L1error / target.var()
    #end
    
    return L1error
#end

class _NormLoss(nn.Module):
    
    def __init__(self, dim_item = 2):
        super(_NormLoss, self).__init__()
        
        self.dim_item = dim_item
    #end
    
    def forward(self, item, mask = None):
        
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


class L2_Loss(nn.Module):
    
    def __init__(self, divide_nitems = False):
        super(L2_Loss, self).__init__()
        
        self.divide_nitems = divide_nitems
    #end
    
    def forward(self, item, mask = None):
        
        if item.__class__ is not torch.Tensor:
            item = torch.Tensor(item)
        #end
        
        if item.shape.__len__() <= 3:
            item = item.unsqueeze(-1)
        #end
        
        if mask is None:
            mask = torch.ones_like(item)
        elif mask.__class__ is not torch.Tensor:
            mask = torch.Tensor(mask)
        #end
        
        argument = item.pow(2)
        argument = argument.mul(mask)
        
        if self.divide_nitems:
            
            if mask.sum() < 1:
                n_items = 1.
            else:
                n_items = mask.sum()
            #end
            
            if n_items < 1:
                raise ValueError('NO DIVISION BY 0 !!!')
            #end
            
            loss = argument.div(n_items)
            loss = torch.sum(loss, dim = (2,3))
            loss = torch.sum(loss, dim = (1,0))
        else:
            loss = argument.mean()
        #end
                
        return loss
    #end
#end

class L1_Loss(nn.Module):
    
    def __init__(self, epsilon = 1e-5, divide_nitems = False):
        super(L1_Loss, self).__init__()
        
        self.epsilon = torch.Tensor([epsilon])
        self.divide_nitems = divide_nitems
    #end
    
    def forward(self, item, mask = None):
        
        if item.__class__ is not torch.Tensor:
            item = torch.Tensor(item)
        #end
        
        if item.shape.__len__() <= 3:
            item = item.unsqueeze(-1)
        #end
        
        if mask is None:
            mask = torch.ones_like(item)
        elif mask.__class__ is not torch.Tensor:
            mask = torch.Tensor(mask)
        #end
        
        argument = torch.sqrt( self.epsilon.pow(2) + item.pow(2) )
        argument = argument.mul(mask)
        
        if self.divide_nitems:
            
            if mask.sum() < 1:
                n_items = 1.
            else:
                n_items = mask.sum()
            #end
            
            if n_items < 1:
                raise ValueError('NO DIVISION BY 0 !!!')
            #end
            
            loss = argument.div(n_items)
            loss = torch.sum(loss, dim = (2,3))
            loss = torch.sum(loss, dim = (1,0))
        else:
            loss = argument.mean()
        #end
                
        return loss
    #end
#end




