
import sys
sys.path.append('../utls')

import numpy as np
import torch
from torch import nn
import futls as fs

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')
#end


###############################################################################
##### CUSTOM TORCH LAYERS #####################################################
###############################################################################

class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()
    #end
    
    def forward(self, signal):
        print(signal.shape)
        return signal
    #end
#end

class FlattenSpatialDim(nn.Module):
    def __init__(self):
        super(FlattenSpatialDim, self).__init__()
    #end
    
    def forward(self, data):
        batch_size, ts_length = data.shape[:2]
        return data.reshape(batch_size, ts_length, -1)
    #end
#end

class Squeeze(nn.Module):
    def __init__(self, squeeze_dim):
        super(Squeeze, self).__init__()
        self.squeeze_dim = squeeze_dim
    #end
    
    def forward(self, data):
        if data.shape[-2] > 1 or data.shape[-1] > 1:
            cx, cy = data.shape[-2] // 2, data.shape[-1] // 2
            return data[:,:,cx,cy].unsqueeze(-1)
        else:
            return data.squeeze(self.squeeze_dim)
        #end
    #end
#end


###############################################################################
##### DEEP LEARNING MODELS ####################################################
###############################################################################


class ConvNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(ConvNet, self).__init__()
        	
        ts_length = shape_data[1] * 3
        
        self.net = nn.Sequential(
            nn.Conv2d(ts_length, 128, (5,5), padding = 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 128, (5,5), padding = 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, ts_length, (5,5), padding = 2)
        )
    #end
    
    def forward(self, data):
        
        reco = self.net(data)
        return reco
    #end
#end


class SConvNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(SConvNet, self).__init__()
        	
        ts_length = shape_data[1] * 3
        
        self.net = nn.Sequential(
            nn.Conv2d(ts_length, 32, (5,5), padding = 2),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(32, ts_length, (5,5), padding = 2)
        )
    #end
    
    def forward(self, data):
        
        reco = self.net(data)
        return reco
    #end
#end


class ConvAutoEncoder(nn.Module):
    def __init__(self, shape_data, config_params):
        super(ConvAutoEncoder, self).__init__()
        
        in_channels = shape_data[1] * 3
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5), padding = 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (3,3), padding = 1),
            nn.AvgPool2d((2,2))
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size = (2,2), stride = 2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(64, in_channels, kernel_size = (3,3), padding = 1)
        )
    #end
    
    def forward(self, data):
        
        code = self.encoder(data)
        reco = self.decoder(code)
        return reco
    #end
#end


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        )
    #end
    
    def forward(self, data):
        return self.conv(data)
    #end
#end

class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor = 4):
        super(Downsample, self).__init__()
        
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(downsample_factor),
            DoubleConv(in_channels, out_channels)
        )
    #end
    
    def forward(self, data):
        return self.down_conv(data)
    #end
#end

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 2, stride = 2)
        )
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size = 5, padding = 2)
    #end
    
    def forward(self, scale1_data, scale2_data):
        
        scale2_data_upscaled = self.up_conv(scale2_data)
        data = torch.cat([scale1_data, scale2_data_upscaled], dim = 1)
        return self.conv(data)
    #end
#end

class UNet1(nn.Module):
    def __init__(self, shape_data, cparams):
        super(UNet1, self).__init__()
        
        in_channels  = shape_data[1] * 3
        out_channels = shape_data[1] * 3 
        self.in_conv = nn.Conv2d(in_channels, in_channels, kernel_size = 5, padding = 2)
        self.down = Downsample(in_channels, 512)
        self.up = Upsample(512, in_channels)
        self.out_conv = nn.Conv2d(in_channels, out_channels, kernel_size = 5, padding = 2)
    #end
    
    def forward(self, data):
        
        x1 = self.in_conv(data)
        x2 = self.down(x1)
        
        x3 = self.up(x1, x2)
        out = self.out_conv(x3)
        return out
    #end
#end

class DepthwiseConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride = 1):
        super(DepthwiseConv2d, self).__init__(
            nn.Conv2d(in_channels, in_channels,
                      kernel_size = kernel_size,
                      padding     = padding,
                      stride      = stride,
                      groups      = in_channels,
                      bias        = True
            ),
            nn.Conv2d(in_channels, 
                      out_channels,
                      kernel_size = 1
            )
        )
    #end
#end

class DoubleConv_pdf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv_pdf, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1)
        )
    #end
    
    def forward(self, data):
        return self.conv(data)
    #end
#end

class Downsample_pdf(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_factor = 4):
        super(Downsample_pdf, self).__init__()
        
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(downsample_factor),
            DoubleConv_pdf(in_channels, out_channels)
        )
    #end
    
    def forward(self, data):
        return self.down_conv(data)
    #end
#end

class Upsample_pdf(nn.Module):
    def __init__(self, in_channels, out_channels, outer_channels, cparams):
        super(Upsample_pdf, self).__init__()
        
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 2, stride = 2),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size = 2, stride = 2)
        )
        self.conv = nn.Conv2d(out_channels * 2, outer_channels, kernel_size = 5, padding = 2)
    #end
    
    def forward(self, scale1_data, scale2_data):
        
        scale2_data_upscaled = self.up_conv(scale2_data)
        data = torch.cat([scale1_data, scale2_data_upscaled], dim = 1)
        return self.conv(data)
    #end
#end

class UNet1_pdf(nn.Module):
    def __init__(self, shape_data, cparams):
        super(UNet1_pdf, self).__init__()
        
        in_channels     = shape_data[1] * 3
        out_channels    = shape_data[1] * 3
        
        # UNet
        self.in_conv    = nn.Conv2d(in_channels, in_channels, kernel_size = 5, padding = 2)
        self.down       = Downsample_pdf(in_channels, 512)
        self.up         = Upsample_pdf(512, in_channels, in_channels, cparams)
        self.out_conv   = nn.Conv2d(in_channels, out_channels, (5,5), padding = (2,2))
    #end
    
    def forward(self, data):
        
        x1  = self.in_conv(data)
        x2  = self.down(x1)
        x3  = self.up(x1, x2)
        out = self.out_conv(x3)
        
        return out
    #end
#end

class HistogrammizationDirect(nn.Module):
    def __init__(self, in_channels, out_channels, shape_data, lr_kernelsize, wind_bins):
        super(HistogrammizationDirect, self).__init__()
        
        self.lr_kernelsize = lr_kernelsize
        self.wind_bins     = wind_bins
        self.nbins         = shape_data[-1]
        self.timesteps     = shape_data[1]
        hist_out_channels  = shape_data[1] * shape_data[-1]
        
        self.conv2d_relu_cascade = nn.Sequential(
            DepthwiseConv2d(in_channels, 256, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            DepthwiseConv2d(256, out_channels, kernel_size = (3,3), padding = 1),
            nn.ReLU(),
            DepthwiseConv2d(out_channels, out_channels, kernel_size = (3,3), padding = 1)
        )
        self.linear_reshape = nn.Conv2d(out_channels, hist_out_channels, kernel_size = 3, padding = 1)
        # self.downsample     = nn.MaxPool2d(lr_kernelsize)
        self.downsample     = nn.AvgPool2d(lr_kernelsize)
        self.relu           = nn.ReLU()
        self.shortcut       = nn.Identity()
        # self.normalize      = nn.LogSoftmax(dim = -1)
        self.normalize      = nn.Softmax(dim = -1)
    #end
    
    def reshape(self, data):
        
        out = torch.zeros(data.shape[0], self.timesteps, *tuple(data.shape[-2:]), self.nbins)
        t_start = 0
        for t in range(self.timesteps-1):
            t_end = self.nbins * t + self.nbins
            out[:,t,:,:,:] = torch.movedim(data[:, t_start : t_end,:,:], 1, 3)
            t_start = t_end
        #end
        
        return out
    #end
    
    def forward(self, data_fields_hr):
        
        # histograms regressor
        out_tmp = self.conv2d_relu_cascade(data_fields_hr.detach())
        out_tmp = self.linear_reshape(out_tmp)
        out_tmp = self.downsample(out_tmp)
        out_tmp = self.reshape(out_tmp)
        
        # HR fields to hist empirical
        fields_emp_hist = data_fields_hr.clone().detach()
        wind_hist_empirical = fs.empirical_histogrammize(fields_emp_hist, 
                                                         self.lr_kernelsize, 
                                                         self.wind_bins,
                                                         laplace_smoothing = True)
        wind_hist_log = torch.log(wind_hist_empirical)
        
        # Residual connection
        out      = out_tmp
        out_res  = out + wind_hist_log
        out_norm = self.normalize(out_res)
        
        return out_norm
    #end
#end

class TrainableFieldsToHist(nn.Module):
    def __init__(self, model, shape_data, cparams):
        super(TrainableFieldsToHist, self).__init__()
        
        in_channels    = shape_data[1] * 1
        out_channels   = 256
        self.timesteps = shape_data[1]
        self.lr_sfreq  = cparams.LR_MASK_SFREQ
        self.Phi       = model
        self.HPhi      = HistogrammizationDirect(in_channels, out_channels, shape_data, cparams.LR_KERNELSIZE, cparams.WIND_BINS)
    #end
    
    def interpolate_lr(self, data_lr, sampling_freq, timesteps = 24):
        return fs.interpolate_along_channels(data_lr, sampling_freq, timesteps)
    #end
    
    def forward(self, data_input, data_gt, normparams):
        
        # Reconstruction of spatial wind speed fields
        fields_ = self.Phi(data_input)
        
        # Interpolate lr part of reconstructions
        # fields_lr_ = self.interpolate_lr(data_input[:,:self.timesteps,:,:], self.lr_sfreq)
        # fields_an_  = fields_[:, 2 * self.timesteps:, :,:]
        # fields_hr_ = fields_an_ + fields_lr_
        fields_hr, fields_lr, fields_an = fs.hr_from_lr_an(fields_, data_input, self.lr_sfreq, self.timesteps)
        fields_hr = fields_hr * normparams['std'] # if not normalize, std is 1
        
        # To histogram
        hist_out  = self.HPhi(fields_hr)
        return hist_out, fields_lr, fields_an
    #end
#end



# TIP
###############################################################################
##### 4DVARNET OBSERVATION MODELS #############################################
###############################################################################


class ModelObs_SM(nn.Module):
    
    def __init__(self, shape_data, dim_obs):
        super(ModelObs_SM, self).__init__()
        
        self.shape_data = shape_data
        self.dim_obs = dim_obs
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
    #end
    
    def forward(self, x, y_obs, mask):
        
        obs_term = (x - y_obs).mul(mask)
        return obs_term
    #end
#end

class ModelObs_MM(nn.Module):
    def __init__(self, shape_data, buoys_coords, dim_obs):
        super(ModelObs_MM, self).__init__()
        
        self.shape_data = shape_data
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        self.buoys_coords = buoys_coords
        self.dim_obs = dim_obs
        in_channels = shape_data[1]
        
        # Conv2d net to produce a lower-dim feature map for latent state
        self.net_state_spatial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
        )
        
        # Same for observations
        self.net_data_spatial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
        )
        
        # Since latent state is a spatial field whatsoever, this model
        # produces a feature map and transforms it in a multi-variate time series
        # to compare with the feature space of the in-situ time series
        self.net_state_situ = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size = (3,3)),
            nn.AvgPool2d((5,5)),
            FlattenSpatialDim(),
            nn.Linear(25,11)
        )
        
        # H situ obs: treating time series (local) so it is Conv1d
        self.net_data_situ = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size = 3),
            nn.LeakyReLU(0.1)
        )
    #end
#end

class ModelObs_MM_mod(ModelObs_MM):
    def __init__(self, shape_data, buoys_coords, dim_obs):
        super(ModelObs_MM_mod, self).__init__(shape_data, buoys_coords, dim_obs)
        
        self.dim_obs = dim_obs
    #end
    
    def forward(self, x, y_obs, mask):
        
        # || x - y ||²
        dy_complete = (x[:,:24] - y_obs[:,:24]).mul(mask[:,:24])
        
        # || h_situ(x) - g_situ(y_situ) ||²
        x_spatial = x[:,:24] + x[:,24:48]
        y_spatial = y_obs[:,:24] + y_obs[:,24:48]
        y_situ = y_spatial[:,:, self.buoys_coords[:,0], self.buoys_coords[:,1]]
        
        # Silence the buoys that are in panne
        for i in range(self.buoys_coords.shape[0]):
            if self.buoys_coords[i,2] == 0:
                y_situ[:,:,i] = 0.
            #end
        #end
        
        feat_state_situ = self.net_state_situ(x_spatial)
        feat_data_situ  = self.net_data_situ(y_situ)
        dy_situ         = (feat_state_situ - feat_data_situ)
        
        # || g_hr(x) - h_hr(y_hr) ||²
        y_spatial = y_spatial.mul(mask[:,24:48])
        feat_state_spatial = self.net_state_spatial(x_spatial)
        feat_data_spatial  = self.net_data_spatial(y_spatial)
        dy_spatial         = (feat_state_spatial - feat_data_spatial)
        
        return [dy_complete, dy_situ, dy_spatial]
    #end
#end

class ModelObs_MM2d(nn.Module):
    def __init__(self, shape_data, dim_obs):
        super(ModelObs_MM2d, self).__init__()
        
        self.shape_data = shape_data
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        timesteps = shape_data[1]
        
        self.net_state = nn.Sequential(
            nn.Conv2d(timesteps, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
        )
        
        self.net_data = nn.Sequential(
            nn.Conv2d(timesteps, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
        )
    #end
    
    def extract_feat_state(self, state):
        
        feat_state = self.net_state(state)
        return feat_state
    #end
    
    def extract_feat_data(self, data):
        
        feat_data = self.net_data(data)
        return feat_data
    #end
#end

class ModelObs_MM2d_mod(ModelObs_MM2d):
    def __init__(self, shape_data, dim_obs):
        super(ModelObs_MM2d_mod, self).__init__(shape_data, dim_obs)
        
        self.dim_obs = dim_obs
    #end
    
    def forward(self, x, y_obs, mask):
        
        # || x - y ||²
        dy_complete = (x[:,:24] - y_obs[:,:24]).mul(mask[:,:24])
        
        # || h(x) - g(y) ||²
        x_spatial = x[:,:24] + x[:,24:48]
        y_spatial = y_obs[:,:24] + y_obs[:,24:48]
        y_spatial = y_spatial.mul(mask[:,24:48])
        feat_data = self.extract_feat_data(y_spatial)
        feat_state = self.extract_feat_state(x_spatial)
        
        dy_spatial = (feat_state - feat_data)
        
        return [dy_complete, dy_spatial]
    #end
#end

class ModelObs_MM1d(nn.Module):
    def __init__(self, shape_data, buoys_coords, dim_obs):
        super(ModelObs_MM1d, self).__init__()
        
        self.dim_obs = dim_obs
        self.shape_data = shape_data
        self.buoys_coords = buoys_coords
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        in_channels = shape_data[1]
        
        self.net_state = torch.nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size = (3,3), padding = 'same'),
            nn.MaxPool2d((5,5)),
            FlattenSpatialDim(),
            nn.Linear(25, 11)
        )
        
        self.net_data = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size = 3),
            nn.LeakyReLU(0.1)
        )
    #end
    
    def extract_feat_state(self, state):
        return self.net_state(state)
    #end
    
    def extract_feat_data(self, data):
        return self.net_data(data)
    #end
#end

class ModelObs_MM1d_mod(ModelObs_MM1d):
    def __init__(self, shape_data, buoys_coords, dim_obs):
        super(ModelObs_MM1d_mod, self).__init__(shape_data, buoys_coords, dim_obs)
        
        self.dim_obs = dim_obs
    #end
    
    def forward(self, x, y_obs, mask):
        
        dy_complete = (x[:,:24] - y_obs[:,:24]).mul(mask[:,:24])
        
        y_spatial = y_obs[:,:24] + y_obs[:,24:48]
        y_situ = y_spatial[:, :, self.buoys_coords[:,0], self.buoys_coords[:,1]]
        x_spatial = x[:,:24] + x[:,24:48]
        
        feat_state = self.extract_feat_state(x_spatial)
        feat_data = self.extract_feat_data(y_situ)
        
        dy_situ = (feat_state - feat_data)
        
        return [dy_complete, dy_situ]
    #end
#end


###############################################################################
##### MODEL SELECTION #########################################################
###############################################################################

def model_selection(shape_data, config_params, normparams = None, components = False):
    
    if config_params.PRIOR == 'SN':
        model = SConvNet(shape_data, config_params)
    #end
    
    elif config_params.PRIOR == 'CN':
        model = ConvNet(shape_data, config_params)
    
    elif config_params.PRIOR == 'AE':
        model = ConvAutoEncoder(shape_data, config_params)
    
    elif config_params.PRIOR == 'UN1':
        model = UNet1(shape_data, config_params)
    
    else:
        raise NotImplementedError('No valid prior')
    #end
    
    return model
    
    # if config_params.VNAME == '4DVN-PDF':
    #     return TrainableFieldsToHist(model, shape_data, config_params)
    # else:
    #     return model
    # #end
#end
