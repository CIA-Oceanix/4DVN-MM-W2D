
import torch
import torch.nn.functional as F
import numpy as np

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    # torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEVICE = torch.device('cpu')
#end


def get_data_mask(shape_data, mask_land, lr_sampling_freq, hr_sampling_freq, hr_obs_points, buoys_positions, mm_obsmodel):
    
    def get_resolution_mask(freq, mask, mask_land, wfreq):
        
        if freq is None:
            return mask
        else:
            if freq.__class__ is list:
                if wfreq == 'lr':
                    mask[:, freq, :,:] = 1.
                elif wfreq == 'hr':
                    mask_land = mask_land.to(mask.device)
                    mask[:, freq, :,:] = mask_land #self.mask_land
                #end
                
            elif freq.__class__ is int:
                for t in range(mask.shape[1]):
                    if t % freq == 0:
                        if wfreq == 'lr':
                            mask[:,t,:,:] = 1.
                        elif wfreq == 'hr':
                            mask_land = mask_land.to(mask.device)
                            mask[:,t,:,:] = mask_land #self.mask_land
                        #end
                    #end
                #end
            #end
        #end
        
        return mask
    #end
    
    # Low-reso pseudo-observations
    mask_lr = torch.zeros(shape_data)
    
    # High-reso dx1 : get according to spatial sampling regime.
    # This gives time series of local observations in specified points
    mask_hr_dx1 = get_mask_HR_observation_points(shape_data, hr_obs_points, buoys_positions)
    mask_hr_dx2 = torch.zeros(shape_data)
    
    mask_lr = get_resolution_mask(lr_sampling_freq, mask_lr, mask_land, 'lr')
    
    if True:
        mask_lr[:,-1,:,:] = 1.
    #end
    
    mask_hr_dx1 = get_resolution_mask(hr_sampling_freq, mask_hr_dx1, mask_land, 'hr')
    
    mask = torch.cat([mask_lr, mask_hr_dx1, mask_hr_dx2], dim = 1)
    return mask, mask_lr, mask_hr_dx1, mask_hr_dx2
#end

def get_mask_HR_observation_points(shape_data, mode, buoys_positions):
    
    if mode.__class__ is list:
        mode = mode[0]
    #end
    
    if mode == 'buoy' or mode == 'buoys':
        
        buoy_coords = buoys_positions
        mask = torch.zeros(shape_data)
        mask[:,:, buoy_coords[:,0], buoy_coords[:,1]] = 1.
        
    elif mode == 'zeroes':
        
        mask = torch.zeros(shape_data)
        
    else:
        raise ValueError('Mask mode not impletemented.')
    #end
    
    return mask
#end

def downsample_and_interpolate_spatially(data, lr_kernel_size):
    
    img_size = data.shape[-2:]
    pooled = F.avg_pool2d(data, kernel_size = lr_kernel_size)
    pooled  = F.interpolate(pooled, size = tuple(img_size), mode = 'bicubic', align_corners = False)
    
    if not data.shape == pooled.shape:
        raise ValueError('Original and Pooled_keepsize data shapes mismatch')
    #end
    
    return pooled
#end

def get_delay_bias(data, lr_sampling_freq, timesteps, timewindow_start, delay_max, delay_min):
    
    batch_size = data.shape[0]
    
    for m in range(batch_size):
        # constant delay for sample (for all timesteps)
        
        # with probability p ...
        p = np.random.uniform(0,1)
        if p > 0.25:
            delay = np.random.randint(delay_min, delay_max)
        else:
            delay = 0
        #end
        
        for t in range(timesteps):
            t_true = t + timewindow_start
            if t_true % lr_sampling_freq == 0:
                try:
                    data[m,t_true,:,:] = data[m,t_true + delay, :,:]
                except:
                    pass
                #end
            #end
        #end
    #end
    
    return data
#end

def get_remodulation_bias(data, lr_sampling_freq, timesteps, timewindow_start, intensity_min, intensity_max):
    
    batch_size = data.shape[0]
    
    """ Alternative :
    if lr_sampling_freq.__class__ is int:
        iteration_list = [i for i in range(timesteps) if i % lr_sampling_freq == 0]
    else:
        iteration_list = lr_sampling_freq
    #end
    """
    
    for m in range(batch_size):
        intensity = np.random.uniform(intensity_min, intensity_max)
        for t in range(timesteps):
            t_true = t + timewindow_start
            if t_true % lr_sampling_freq == 0:
                try:
                    data[m,t_true,:,:] = data[:,t_true,:,:] * intensity
                except:
                    pass
                #end
            #end
        #end
    #end
    
    return data
#end

def interpolate_along_channels(data, sampling_freq, timesteps):
    
    img_shape = data.shape[-2:]
    
    # Isolate timesteps related to LR data
    data_to_interpolate = torch.zeros((data.shape[0], timesteps // sampling_freq + 1, *img_shape))
    for t in range(timesteps):
        if t % sampling_freq == 0:
            data_to_interpolate[:, t // sampling_freq, :,:] = torch.Tensor(data[:,t,:,:])
        #end
    #end
    data_to_interpolate[:,-1,:,:] = torch.Tensor(data[:,-1,:,:] )
    
    # Interpolate channel-wise (that is, timesteps)
    data_interpolated = F.interpolate(data_to_interpolate.permute(0,2,3,1), 
                                      [img_shape[0], timesteps], 
                                      mode = 'bicubic', align_corners = False)
    data_interpolated = data_interpolated.permute(0,3,1,2)
    
    return data_interpolated
#end

def get_persistency_model(data, frequency):
    
    persistence = torch.zeros(data.shape)
    
    if frequency is None:
        pass
    elif frequency.__class__ is int:
        t_previous = 0
        for t in range(data.shape[1]):
            if t % frequency == 0:
                persistence[:,t,:,:] = data[:,t,:,:]
                t_previous = t
            else:
                persistence[:,t,:,:] = data[:,t_previous,:,:]
            #end
        #end
    elif frequency.__class__ is list:
        t_previous = 0
        for p in range(frequency.__len__()):
            try:
                t_next = (frequency[p] + frequency[p + 1]) // 2
            except:
                t_next = data.shape[1]
            #end
            for t in range(t_previous, t_next):
                persistence[:,t,:,:] = data[:, frequency[p], :,:]
            #end
            t_previous = t_next
        #end
    #end

    return persistence
#end

