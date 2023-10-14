
import torch
from torch import nn
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
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

class HellingerDistance(nn.Module):
    def __init__(self, n_items = False):
        super(HellingerDistance, self).__init__()
        
        self.n_items = n_items
    #end
    
    def forward(self, target, output):
        '''
        Inputs: normalized probabilities, ie histograms summing to 1!!!
        
        Hellinger distance: metric to compare proximity between histograms.
        Given two probability distributions :math:`P` and :math:`Q`, is computed as 
        
            .. math::
                HD(P,Q) = \sqrt{1 - \sum_{x \in X} \sqrt{P(x) Q(x)}}
        
        with :math:`\sum_x P(x) = 1` and :math:`\sum_x Q(x) = 1`.
        '''
        
        b_coefficient = torch.sqrt( target * output ).sum(-1)
        b_coefficient[b_coefficient > 1] = 1.
        hellinger_distance = torch.sqrt(1. - b_coefficient)
        
        return hellinger_distance.mean()
    #end
#end

class KLDivLoss(torch.nn.Module):
    def __init__(self):
        super(KLDivLoss, self).__init__()
    #end
    
    def forward(self, output, target, log_target = False):
        
        if target.__class__ is not torch.Tensor:
            target = torch.Tensor(target)
        if output.__class__ is not torch.Tensor:
            output = torch.Tensor(output)
        #end
        
        batch_size, timesteps, height, width, bins = target.shape
        target = target.reshape(batch_size * timesteps * height * width, bins)
        output = output.reshape(batch_size * timesteps * height * width, bins)
        
        target[target < 1e-6] = 1e-6
        if not log_target:
            kld = target * (target.log() - output)
        else:
            kld = target.exp() * (target - output)
        #end
        
        # kld = kld.sum(-1)
        # kld = kld.mean(dim = (-2, -1))
        # kld = kld.sum(-1)
        return kld.sum(-1).mean()
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




