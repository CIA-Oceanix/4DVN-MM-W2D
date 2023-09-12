
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
#end


def get_resolution_mask_(freq, mask, mask_land, wfreq):
    
    if freq is None:
        return mask
    else:
        if freq.__class__ is list:
            if wfreq == 'lr':
                mask[:, freq, :,:] = 1.
            elif wfreq == 'hr':
                mask_land = mask_land.to(mask.device)
                mask[:, freq, :,:] = mask_land
            #end
            
        elif freq.__class__ is int:
            for t in range(mask.shape[1]):
                if t % freq == 0:
                    if wfreq == 'lr':
                        mask[:,t,:,:] = 1.
                    elif wfreq == 'hr':
                        mask_land = mask_land.to(mask.device)
                        mask[:,t,:,:] = mask_land
                    #end
                #end
            #end
        #end
    #end
    
    return mask
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
                    mask[:, freq, :,:] = mask_land
                #end
                
            elif freq.__class__ is int:
                for t in range(mask.shape[1]):
                    if t % freq == 0:
                        if wfreq == 'lr':
                            mask[:,t,:,:] = 1.
                        elif wfreq == 'hr':
                            mask_land = mask_land.to(mask.device)
                            mask[:,t,:,:] = mask_land
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
    mask_hr_dx1 = get_resolution_mask(hr_sampling_freq, mask_hr_dx1, mask_land, 'hr')
    
    mask_hr_dx2 = torch.zeros(shape_data)
    
    mask_lr = get_resolution_mask(lr_sampling_freq, mask_lr, mask_land, 'lr')
    
    if True:
        # Artificially the last observation is set equal to the 
        # first of the next day, so to have a last datum in the timeseries
        mask_lr[:,-1,:,:] = 1.
    #end
    
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
        
        # ugly as fuck but whatever works
        for i in range(buoy_coords.shape[0]):
            if buoy_coords[i,2] == 1:
                mask[:,:, buoy_coords[i,0], buoy_coords[i,1]] = 1.
            else:
                mask[:,:, buoy_coords[i,0], buoy_coords[i,1]] = 0.
            #end
        #end
        
    elif mode == 'zeroes':
        
        mask = torch.zeros(shape_data)
        
    else:
        raise ValueError('Mask mode not impletemented.')
    #end
    
    return mask.to(DEVICE)
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

def get_histogram(data, bins, to_beaufort_scale = False, histogrammization_op = 'pytorch'):
    
    if histogrammization_op == 'handcrafted':
        histogram = torch.zeros(bins.shape[0] - 1)
        for eidx in range(bins.__len__() - 1):
            histogram[eidx] = torch.numel(data[(data > bins[eidx]) & (data <= bins[eidx + 1])])
        #end
    elif histogrammization_op == 'pytorch':
        histogram = torch.histogram(data, bins = bins)[0]
    else:
        raise NotImplementedError('Not implemented. Likely mistyped')
    #end
    
    return histogram
#end

def make_hist(data_, bins, normalized = True, histogrammization_op = 'pytorch'):
    
    if bins.__class__ is not torch.Tensor:
        bins = torch.Tensor(bins)
    #end
    
    # h = get_histogram(data_, bins = bins)
    # h = torch.histogram(data_, bins = bins)[0]
    h = get_histogram(data_, bins, histogrammization_op = histogrammization_op)
    h = torch.autograd.Variable(h)
    
    if normalized:
        return h.div(h.sum())
    else:
        return h
    #end
#end

def fieldsHR2hist(data_field, kernel_size, bins, progbars = False, verbose = True):
    '''
    Takes as input tensors of dimension
        (batch_size, timesteps, height, width)
    '''
    if verbose:
        print('EMPIRICAL TO HISTOGRAM')
    #end

    def lr_dim(dim, ks_):
        return np.int32(np.floor( (dim - ks_) / ks_ + 1 ))
    #end
    
    batch_size, timesteps, heigth, width = data_field.shape
    height_lr, width_lr = lr_dim(heigth, kernel_size), lr_dim(width, kernel_size)
    data_hist = torch.zeros((batch_size, timesteps, height_lr, width_lr, bins.__len__() - 1))
    
    # loop to prepare histogram data
    
    if progbars:
        progbar_batches   = tqdm(range(batch_size), desc = 'Batches     ', position = 0, leave = True)
    else:
        progbar_batches   = range(batch_size)
    #end
    progbar_timesteps = range(timesteps)
    progbar_height    = range(height_lr)
    progbar_width     = range(width_lr)
    
    for m in progbar_batches:
        for t in progbar_timesteps:
            
            i_start = 0
            for i in progbar_height:
                
                j_start = 0
                for j in progbar_width:
                    
                    i_end = i_start + kernel_size
                    j_end = j_start + kernel_size
                    
                    try:
                        this_wind_pixel = data_field[m,t, i_start:i_end ,j_start:j_end]
                    except:
                        this_wind_pixel = data_field[m,t, i_start:, j_start:]
                    #end
                    
                    hist = make_hist(this_wind_pixel, bins, histogrammization_op = 'handcrafted')
                    data_hist[m,t,i,j,:] = hist
                    
                    # if torch.any(data_hist.isnan()):
                    #     print()
                    #     print('Nans')
                    # #end
                    
                    j_start = j_end
                #end
                
                i_start = i_end
            #end
        #end
    #end
    
    return data_hist
#end

def fieldsHR2hist_V1(data_field, kernel_size, bins, progbars = False):
    '''Gives the same as fieldsHR2hist, is a bit more elegant and sophisticated but slower.
    Stated otherwise, is almost useless, mannaggia alla curia'''

    def split(array, nrows, ncols):
        '''
        Split a matrix into sub-matrices.
        Thanks: https://stackoverflow.com/questions/11105375/how-to-split-a-matrix-into-4-blocks-using-numpy
        NOTE: nrows and ncols are the number of rows and columns of EACH SUB-BLOCK, not the number of 
        rows and columns in which the main matrix is decomposed!!!
        '''
        
        r, h = array.shape
        return (array.reshape(h//nrows, nrows, -1, ncols).swapaxes(1, 2).reshape(-1, nrows, ncols))
    #end
    
    batch_size, timesteps, height_hr, width_hr = data_field.shape
    downsampled_img_shape = tuple((np.int32(height_hr // kernel_size), np.int32(width_hr // kernel_size)))
    data_field = data_field.reshape(-1, height_hr, width_hr)

    fields_list = [data_field[t,:,:] for t in range(batch_size * timesteps)]
    histograms = torch.zeros(batch_size, timesteps, *downsampled_img_shape, bins.__len__()-1).reshape(-1, *downsampled_img_shape, bins.__len__()-1)

    for t in tqdm(range(batch_size * timesteps), desc = 'Batches x Timesteps : ', position = 0, leave = True):

        field_splitted = split(fields_list[t], kernel_size, kernel_size)

        for n in range(field_splitted.shape[0]):
            # PAY CLOSE ATTENTION TO THIS:
            # divmod has as first argument the SEQUENTIAL index of the sub-blocks collection
            # and as second argument the number of COLUMNS. Since it has to moduloize according
            # to the width of the array
            i,j = divmod(n, downsampled_img_shape[1])
            h = make_hist(field_splitted[n,:,:].flatten(), bins = bins, normalized = True, histogrammization_op = 'pytorch')
            histograms[t,i,j,:] = h
        #end
    #end

    histograms = histograms.reshape(batch_size, timesteps, *downsampled_img_shape, bins.__len__()-1)
    return histograms
#end

def fieldsLR2hist(data_field, bins):
    
    batch_size, timesteps, height, width = data_field.shape
    data_hist = torch.zeros((batch_size, timesteps, height, width,  bins.__len__() - 1))
    
    if True:
        progbar_batches   = tqdm(range(batch_size), desc = 'Batches     ', position = 0, leave = True)
        progbar_timesteps = tqdm(range(timesteps),  desc = 'Timesteps   ', position = 1, leave = False)
        progbar_height    = tqdm(range(height),     desc = 'Loop i (lr) ', position = 2, leave = False)
        progbar_width     = tqdm(range(width),      desc = 'Loop j (lr) ', position = 3, leave = False)
    else:
        progbar_batches   = range(batch_size)
        progbar_timesteps = range(timesteps)
        progbar_height    = range(height)
        progbar_width     = range(width)
    #end
    
    for m in progbar_batches:
        for t in progbar_timesteps:
            
            for i in progbar_height:
                for j in progbar_width:
                    
                    hist = make_hist(data_field[m,t,i,j], bins)
                    data_hist[m,t,i,j,:] = hist
                #end
            #end
        #end
    #end
    
    return data_hist
#end

