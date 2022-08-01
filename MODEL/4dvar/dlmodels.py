import torch
from torch import nn
import pytorch_lightning as pl


class Phi_r(nn.Module):
    ''' Dynamical prior '''
    
    def __init__(self, model):
        super(Phi_r, self).__init__()
        
        self.net = model
    #end
    
    def forward(self, data):
        
    	return self.net(data)
    #end
#end


class ObservationModel(torch.nn.Module):
    ''' Observation model : mask observations '''
    
    def __init__(self):
    	super(ObservationModel, self).__init__()
    	
    	pass
    #end
    
    def forward(self, y_obs, x):
    	
    	pass
    #end
#end


class LitModel(pl.LightningModule):
    
    def __init__(self, Phi, shape_data):
        super(LitModel, self).__init__()
        
        
        self.Phi = Phi
    #end
    
    def forward(self, data):
        
        loss, out = self.compute_loss(data)
        return loss, out
    #end
    
    def configure_optimizers(self):
        
        optimizers = None
        return optimizers
    #end
    
    def compute_loss(self, data, phase = 'train'):
        
        loss = None
        out = None
        return loss, out
    #end
    
    def training_step(self):
        
        pass
    #end
    
    def training_epoch_end(self):
        
        pass
    #end
    
    def validation_step(self):
        
        pass
    #end
    
    def validation_epoch_end(self):
        
        pass
    #end
    
    def test_step(self):
        
        pass
    #end
    
#end
