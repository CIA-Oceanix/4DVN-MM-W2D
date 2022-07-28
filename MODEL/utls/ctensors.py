
import torch

class CTensor(torch.Tensor):
    
    def __init__(self, data, dtype = torch.float32):
        super(CTensor, self).__init__()
        
        self.data = torch.Tensor(data)
    #end
    
    def remove_nans(self):
        
        self.data[self.data.isnan()] = 0.
    #end
    
    def get_mask(self, remove_nans = True):
        
        mask = torch.zeros_like(self.data)
        mask[self.data.isnan().logical_not()] = 1.
        mask[self.data == 0] = 0
        
        if remove_nans:
            self.remove_nans()
        #end
        
        return mask
    #end
    
    def get_nitem(self):
        
        self_mask = self.get_mask()
        num_features = self_mask.shape[-1]
        return self_mask.sum().div(num_features)
    #end
    
    def forward(self):
        
        return self
    #end
#end

