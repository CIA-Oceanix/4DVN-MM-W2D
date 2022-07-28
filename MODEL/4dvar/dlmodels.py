import torch
from torch import nn
import pytorch_lightning as pl


class Phi_r(nn.Module):
    
    def __init__(self):
        super(Phi_r, self).__init__()
        
        self.net = None
    #end
    
    def forward(self, data):
    
    	return self.net(data)
    #end
#end


class ObservationModel(torch.nn.Module):
    
    def __init__(self):
    	super(ObservationModel, self).__init__()
    	
    	pass
    #end
    
    def forward(self, y_obs, x):
    	
    	pass
    #end
#end


class LitModel(pl.LightningModule):
    
    def __init__(self):
        super(LitModel, self).__init__()
        
    pass
#end
