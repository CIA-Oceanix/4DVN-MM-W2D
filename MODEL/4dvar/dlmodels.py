
import sys
sys.path.append('../utls')

import numpy as np
import torch
from torch import nn
from unet import UNet4

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

class dw_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, groups, bias):
        super(dw_conv2d, self).__init__()
        
        self.conv_depthwise = nn.Conv2d(in_channels, in_channels,
                                        kernel_size = kernel_size,
                                        padding = padding,
                                        stride = stride,
                                        groups = in_channels,
                                        bias = bias)
        self.conv_pointwise = nn.Conv2d(in_channels, out_channels,
                                        kernel_size = 1,
                                        bias = bias)
    #end
    
    def forward(self, data):
        
        spatial_out = self.conv_depthwise(data)
        output = self.conv_pointwise(spatial_out)
        return output
    #end
#end

class RBlock(nn.Module):
    
    def __init__(self):
        super(RBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(72, 50, (3,3), padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(0.1),
            nn.Conv2d(50, 72, (3,3), padding = 1, stride = 1),
        )
        
        self.shortcut = nn.Identity()
    #end
    
    def forward(self, data):
        
        out = self.block(data)
        out = out.add(self.shortcut(data))
        return out
    #end
#end

class ResNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(ResNet, self).__init__()
        
        self.rnet = nn.Sequential(
            RBlock()
        )
    #end
    
    def forward(self, data):
        
        return self.rnet(data)
    #end
#end

class CBlock(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                      padding = 'same',
                      padding_mode = 'reflect',
                      bias = True),
            # nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
    #end
#end

class ConvNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(ConvNet, self).__init__()
        	
        ts_length = shape_data[1] * 3
        
        self.net = nn.Sequential(
            CBlock(ts_length, 32, 5, 2),
            nn.Conv2d(32, ts_length, (5,5),
                      padding = 'same',
                      padding_mode = 'reflect',
                      bias = True)
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

class UNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(UNet, self).__init__()
        
        ts_length = shape_data[1] * 3
        
        self.encoder1 = nn.Conv2d(ts_length, 32, kernel_size = 5, padding = 2)
        self.nl1 = nn.LeakyReLU(0.1)
        self.bottleneck = nn.Conv2d(32, 32, kernel_size = 5, padding = 2)
        self.nl2 = nn.LeakyReLU(0.1)
        self.decoder1 = nn.Conv2d(32 * 2, 32, kernel_size = 5, padding = 2)
        self.conv = nn.Conv2d(32, ts_length, kernel_size = 5, padding = 2)
    #end
    
    def forward(self, x):
        
        enc1 = self.nl1(self.encoder1(x))
        bottleneck = self.bottleneck(enc1)
        dec1 = torch.cat([enc1, bottleneck], dim = 1)
        dec1 = self.decoder1(dec1)
        y = self.conv(self.nl2(dec1))
        return y
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
    def __init__(self, in_channels, out_channels):
        super(UNet1, self).__init__()
        
        self.in_conv = nn.Conv2d(in_channels, in_channels, kernel_size = 5, padding = 2)
        self.down = Downsample(in_channels, 128)
        self.up = Upsample(128, in_channels)
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



###############################################################################
##### 4DVARNET OBSERVATION MODELS #############################################
###############################################################################

class ModelObs_base(nn.Module):
    def __init__(self, shape_data, dim_obs):
        super(ModelObs_base, self).__init__()
        
        self.shape_data = shape_data
        self.dim_obs    = dim_obs
    #end
#end


class ModelObs_SM(nn.Module):
    ''' Observation model '''
    
    def __init__(self, shape_data, wind_modulus, dim_obs):
        super(ModelObs_SM, self).__init__()
        
        # NOTE : chennels == time series length
        self.shape_data = shape_data
        self.dim_obs = dim_obs
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        self.wind_modulus = wind_modulus
    #end
    
    def forward(self, x, y_obs, mask):
        
        if self.wind_modulus:
            obs_term = (x - y_obs).mul(mask)
            return obs_term
        else:
            obs_term = (x - y_obs).mul(mask)
            return obs_term
        #end
    #end
#end


class ModelObs_MM(nn.Module):
    def __init__(self, shape_data, buoys_coords, wind_modulus, dim_obs):
        super(ModelObs_MM, self).__init__()
        
        self.dim_obs    = dim_obs
        self.shape_data = shape_data
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        self.buoys_coords = buoys_coords
        self.wind_modulus = wind_modulus
        timesteps       = shape_data[1]
        in_channels     = timesteps
        
        # H situ state: same structure. The state is a 2D tensor either way
        self.net_state_Hhr = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size = (3,3)),
            nn.AvgPool2d((5,5)),
            FlattenSpatialDim(),
            nn.Linear(25,11)
        )
        
        self.net_state_Hsitu = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size = (3,3)),
            nn.AvgPool2d((5,5)),
            FlattenSpatialDim(),
            nn.Linear(25,11)
        )
        
        # H situ obs: treating time series (local) so it is Conv1d
        self.net_data_Hsitu = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size = 3),
            nn.LeakyReLU(0.1)
        )
        
        # H hr obs: take the (rare) 2D fields so shares the same structure as for Hmm2d
        self.net_data_Hhr = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, kernel_size = (3,3)),
            nn.MaxPool2d((5,5)),
            nn.LeakyReLU(0.1),
            FlattenSpatialDim(),
            nn.Linear(25,11)
        )
    #end
    
    def extract_feat_state_Hhr(self, state):
        
        feat_state = self.net_state_Hhr(state)
        return feat_state
    #end
    
    def extract_feat_state_Hsitu(self, state):
        
        feat_state = self.net_state_Hsitu(state)
        return feat_state
    #end
    
    def extract_feat_data_Hhr(self, data):
        
        feat_data = self.net_data_Hhr(data)
        return feat_data
    #end
    
    def extract_feat_data_Hsitu(self, data):
        
        feat_data = self.net_data_Hsitu(data)
        return feat_data
    #end
    
    def forward(self, x, y_obs, mask):
        
        if self.wind_modulus:
            # || x - y ||²
            dy_complete = (x[0] - y_obs[0]).mul(mask[0])
            
            # || h_situ(x) - g_situ(y_situ) ||²
            y_situ = y_obs[1][:,:, self.buoys_coords[:,0], self.buoys_coords[:,1]]
            x_situ = x[0] + x[1]
            
            feat_state_situ = self.extract_feat_state_Hsitu(x_situ)
            feat_data_situ  = self.extract_feat_data_Hsitu(y_situ)
            dy_situ         = (feat_state_situ - feat_data_situ)
            
            # || g_hr(x) - h_hr(y_hr) ||²
            y_spatial = y_obs[1].mul(mask[1])
            feat_state_spatial = self.extract_feat_state_Hhr(x[1])
            feat_data_spatial  = self.extract_feat_data_Hhr(y_spatial)
            dy_spatial         = (feat_state_spatial - feat_data_spatial)
            
            return [dy_complete, dy_situ, dy_spatial]
            
        else:
            
            data_dim = self.shape_data[-2:]
            
            # || x - y ||² (two components, low-resolution)            
            dy_lr_u = (x[0][:,:,:, :data_dim[1]] - y_obs[0][:,:,:, :data_dim[1]]).mul(mask[0])
            dy_lr_v = (x[0][:,:,:, -data_dim[1]:] - y_obs[0][:,:,:, -data_dim[1]:]).mul(mask[0])
            
            # || g(x) - h(y) ||²
            
            ## Spatial
            ## Here we need the wind modulus of y and x
            ## x (high-reso) = x (low-reso) + anomaly du
            mask_hr = mask[1]
            
            x_hr_u = x[0][:,:,:, :data_dim[1]] + x[1][:,:,:, :data_dim[1]]
            x_hr_v = x[0][:,:,:, -data_dim[1]:] + x[1][:,:,:, -data_dim[1]:]
            y_hr_u = y_obs[1][:,:,:, :data_dim[1]]
            y_hr_v = y_obs[1][:,:,:, -data_dim[1]:]
            
            x_hr_spatial = torch.autograd.Variable((x_hr_u.pow(2) + x_hr_v.pow(2)).sqrt())
            y_hr_spatial = torch.autograd.Variable((y_hr_u.pow(2) + y_hr_v.pow(2)).sqrt().mul(mask_hr))
            
            feat_state_spatial = self.extract_feat_state_Hhr(x_hr_spatial)
            feat_data_spatial  = self.extract_feat_data_Hhr(y_hr_spatial)
            dy_hr_spatial      = (feat_state_spatial - feat_data_spatial)
            
            ## Situ
            ## Here we isolate the in-situ time series from the hr fields
            y_situ = y_hr_spatial[:,:, self.buoys_coords[:,0], self.buoys_coords[:,1]]
            feat_state_situ = self.extract_feat_state_Hsitu(x_hr_spatial)
            feat_data_situ  = self.extract_feat_data_Hsitu(y_situ)
            dy_hr_situ      = (feat_state_situ - feat_data_situ)
            
            return [dy_lr_u, dy_lr_v, dy_hr_spatial, dy_hr_situ]
        #end
    #end
#end


class ModelObs_MM2d(nn.Module):
    def __init__(self, shape_data, wind_modulus, dim_obs):
        super(ModelObs_MM2d, self).__init__()
        
        self.dim_obs = dim_obs
        self.shape_data = shape_data
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        self.wind_modulus = wind_modulus
        timesteps = shape_data[1]
        
        self.net_state = nn.Sequential(
            nn.Conv2d(timesteps, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
            # nn.Conv2d(128, 64, kernel_size = (3,3)),
            # Squeeze(-1)
        )
        
        self.net_data = nn.Sequential(
            nn.Conv2d(timesteps, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
            # nn.Conv2d(128, 64, kernel_size = (3,3)),
            # Squeeze(-1)
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
    
    def forward(self, x, y_obs, mask):
        
        if self.wind_modulus:
            # || x - y ||²
            dy_complete = (x[0] - y_obs[0]).mul(mask[0])
            
            # || h(x) - g(y) ||²
            x_spatial = x[0] + x[1]
            feat_data = self.extract_feat_data(y_obs[1].mul(mask[1]))
            feat_state = self.extract_feat_state(x_spatial)
            
            dy_spatial = (feat_state - feat_data)
            
            return [dy_complete, dy_spatial]
        
        else:
            
            data_dim = self.shape_data[-2:]
            
            # || x - y ||² (two components, low-resolution)           
            dy_lr_u = (x[0][:,:,:, :data_dim[1]] - y_obs[0][:,:,:, :data_dim[1]]).mul(mask[0])
            dy_lr_v = (x[0][:,:,:, -data_dim[1]:] - y_obs[0][:,:,:, -data_dim[1]:]).mul(mask[0])
            
            # || g(x) - h(y) ||²
            
            ## Spatial
            ## Here we need the wind modulus of y and x
            ## x (high-reso) = x (low-reso) + anomaly du
            mask_hr = mask[1]
            
            x_hr_u = x[0][:,:,:, :data_dim[1]] + x[1][:,:,:, :data_dim[1]]
            x_hr_v = x[0][:,:,:, -data_dim[1]:] + x[1][:,:,:, -data_dim[1]:]
            y_hr_u = y_obs[1][:,:,:, :data_dim[1]]
            y_hr_v = y_obs[1][:,:,:, -data_dim[1]:]
            
            x_hr_spatial = torch.autograd.Variable((x_hr_u.pow(2) + x_hr_v.pow(2)).sqrt())
            y_hr_spatial = torch.autograd.Variable((y_hr_u.pow(2) + y_hr_v.pow(2)).sqrt().mul(mask_hr))
            
            feat_state_spatial = self.extract_feat_state(x_hr_spatial)
            feat_data_spatial  = self.extract_feat_data(y_hr_spatial)
            dy_hr_spatial      = (feat_state_spatial - feat_data_spatial)
            
            return [dy_lr_u, dy_lr_v, dy_hr_spatial]
        #end
    #end
#end


class ModelObs_MM1d(nn.Module):
    def __init__(self, shape_data, buoys_coords, wind_modulus, dim_obs):
        super(ModelObs_MM1d, self).__init__()
        
        self.dim_obs = dim_obs
        self.shape_data = shape_data
        self.buoys_coords = buoys_coords
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        self.wind_modulus = wind_modulus
        timesteps = shape_data[1]
        in_channels = timesteps
        
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
    
    def forward(self, x, y_obs, mask):
        
        if self.wind_modulus:
            dy_complete = (x[0] - y_obs[0]).mul(mask[0])
            
            y_situ = y_obs[1][:, :, self.buoys_coords[:,0], self.buoys_coords[:,1]]
            x_hr_spatial = x[0] + x[1]
            
            feat_state = self.extract_feat_state(x_hr_spatial)
            feat_data = self.extract_feat_data(y_situ)
            
            dy_situ = (feat_state - feat_data)
            
            return [dy_complete, dy_situ]
        
        else:
            
            data_dim = self.shape_data[-2:]
            
            # || x - y ||² (two components, low-resolution)
            
            mask_lr = mask[0]
            
            y_lr_u = y_obs[0][:,:,:, :data_dim[1]]
            y_lr_v = y_obs[0][:,:,:, -data_dim[1]:]
            x_lr_u = x[0][:,:,:, :data_dim[1]]
            x_lr_v = x[0][:,:,:, -data_dim[1]:]
            
            dy_lr_u = (x_lr_u - y_lr_u).mul(mask_lr)
            dy_lr_v = (x_lr_v - y_lr_v).mul(mask_lr)
            
            # || g(x) - h(y) ||²
            
            ## Spatial
            ## Here we need the wind modulus of y and x
            ## x (high-reso) = x (low-reso) + anomaly du
            mask_hr = mask[1]
            
            x_hr_u = x_lr_u + x[1][:,:,:, :data_dim[1]]
            x_hr_v = x_lr_v + x[1][:,:,:, -data_dim[1]:]
            y_hr_u = y_obs[1][:,:,:, :data_dim[1]]
            y_hr_v = y_obs[1][:,:,:, -data_dim[1]:]
            
            x_hr_spatial = (x_hr_u.pow(2) + x_hr_v.pow(2)).sqrt()
            y_hr_spatial = (y_hr_u.pow(2) + y_hr_v.pow(2)).sqrt().mul(mask_hr)
            
            x[1] = torch.cat([x_hr_u, x_hr_v], dim = -1)
            
            ## Situ
            ## Here we isolate the in-situ time series from the hr fields
            y_situ = y_hr_spatial[:,:, self.buoys_coords[:,0], self.buoys_coords[:,1]]
            feat_state_situ = self.extract_feat_state(x_hr_spatial)
            feat_data_situ  = self.extract_feat_data(y_situ)
            dy_hr_situ      = (feat_state_situ - feat_data_situ)
            
            return [dy_lr_u, dy_lr_v, dy_hr_situ]
        #end
    #end
#end


###############################################################################
##### MODEL SELECTION #########################################################
###############################################################################

def model_selection(shape_data, config_params):
    
    if config_params.PRIOR == 'SN':
        return ConvNet(shape_data, config_params)
    elif config_params.PRIOR == 'RN':
        return ResNet(shape_data, config_params)
    elif config_params.PRIOR == 'UN':
        return UNet(shape_data, config_params)
    elif config_params.PRIOR == 'UN1':
        return UNet1(shape_data[1] * 3, shape_data[1] * 3)
    elif config_params.PRIOR == 'UN4':
        return UNet4(shape_data[1] * 3, shape_data[1] * 3)
    else:
        raise NotImplementedError('No valid prior')
    #end
#end
