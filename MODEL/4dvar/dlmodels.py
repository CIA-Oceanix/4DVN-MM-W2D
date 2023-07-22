
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

class MLP(nn.Module):
    def __init__(self, cparams, shape_data):
        super(MLP, self).__init__()
        
        batch_size, ts_length, dim_h, dim_w = shape_data
        if not cparams.WIND_MODULUS:
            dim_w *= 2
        #end
        self.net = nn.Sequential(
            nn.Linear(dim_h * dim_w, 200),
            nn.LeakyReLU(0.1),
            nn.Linear(200, dim_h * dim_w)
        )
    #end
    
    def forward(self, data):
        
        batch_size, ts_length, dim_h, dim_w = data.shape
        data_flat = data.reshape(batch_size, ts_length, dim_h * dim_w)
        reco_flat = self.net(data_flat)
        reco = reco_flat.reshape(batch_size, ts_length, dim_h, dim_w)
        return reco
    #end
#end


class ResNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(ResNet, self).__init__()
        
        self.rnet = nn.Sequential(
            nn.Conv2d(72, 50, (3,3), padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(50),
            nn.LeakyReLU(0.1),
            nn.Conv2d(50, 72, (3,3), padding = 1, stride = 1)
        )
        
        self.shortcut = nn.Identity()
    #end
    
    def forward(self, data):
        
        out = self.rnet(data)
        out = out.add(self.shortcut(data))
        return out
    #end
#end

class CBlock(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(CBlock, self).__init__(
            nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), 
                       padding = 'same',
                       padding_mode = 'reflect',
                      bias = True),
            nn.LeakyReLU(0.1)
        )
    #end
#end

class ConvNet(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(ConvNet, self).__init__()
        	
        ts_length = shape_data[1] * 3
        
        self.net = nn.Sequential(
            nn.Conv2d(ts_length, 32, (5,5), padding = 2),
            nn.Conv2d(32, ts_length, (5,5), padding = 2)
        )
    #end
    
    def forward(self, data):
        
        reco = self.net(data)
        return reco
    #end
#end

class ConvNet_mwind_angle(nn.Module):
    
    def __init__(self, shape_data, config_params):
        super(ConvNet_mwind_angle, self).__init__()
        
        ts_length = shape_data[1] * 3
        input_planes = ts_length * 3
        
        self.net = nn.Sequential(
            nn.Conv2d(input_planes, 64, (5,5), padding = 2),
            nn.Conv2d(64, input_planes, (5,5), padding = 2)
        )
        self.nonlinearity = nn.Tanh()
    #end
    
    def forward(self, data):
        
        output = self.net(data)
        output[:,72:,:,:] = self.nonlinearity(output[:,72:,:,:])
        return output
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
        self.nl1 = nn.Identity()
        self.bottleneck = nn.Conv2d(32, 32, kernel_size = 5, padding = 2)
        # self.nl2 = nn.LeakyReLU(0.1)
        self.nl2 = nn.Identity()
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

class UNet_pdf(nn.Module):
    def __init__(self,shape_data, config_params):
        super(UNet_pdf, self).__init__()
        
        in_channels = shape_data[1] * shape_data[-1]
        self.encoder1 = nn.Conv2d(in_channels, 32, kernel_size = 5, padding = 2)
        self.nl1 = nn.LeakyReLU(0.1)
        self.bottleneck = nn.Conv2d(32, 32, kernel_size = 5, padding = 2)
        self.nl2 = nn.LeakyReLU(0.1)
        self.decoder1 = nn.Conv2d(32 * 2, 32, kernel_size = 5, padding = 2)
        self.conv = nn.Conv2d(32, in_channels, kernel_size = 5, padding = 2)
        self.softmax = nn.Softmax(dim = -1)
    #end
    
    def forward(self, x):
        
        batch_size, timesteps, height, width, bins = x.shape
        x_in = x.reshape(batch_size, timesteps * bins, height, width)
        enc1 = self.nl1(self.encoder1(x_in))
        bottleneck = self.bottleneck(enc1)
        dec1 = torch.cat([enc1, bottleneck], dim = 1)
        dec1 = self.decoder1(dec1)
        y = self.conv(self.nl2(dec1))
        out = torch.reshape(y, (batch_size, timesteps, height, width, bins))
        out = self.softmax(out)
        
        return out
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


class FokkerPlankNet(nn.Module):
    def __init__(self, shape_data, config_params):
        super(FokkerPlankNet, self).__init__()
        
        batch_size, timesteps, height, width, hbins = shape_data
        self.shape_data = (timesteps, height, width,hbins)
        self.net_hist = nn.Sequential(
            nn.Conv2d(timesteps, timesteps * 2, kernel_size = (5,3), padding = (2,1)),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(timesteps * 2, timesteps * 2, kernel_size = (5,3), padding = (2,1)),
            nn.Conv2d(timesteps * 2, timesteps, kernel_size = (5,3), padding = (2,1)),
            nn.Softmax(dim = -1)
        )
        
        self.net_lr = nn.Sequential(
            nn.Conv2d(timesteps, timesteps, kernel_size = 3, padding = 1),
            # nn.LeakyReLU(0.1),
            nn.Conv2d(timesteps, timesteps, kernel_size = 3, padding = 1),
        )
    #end
    
    def forward(self, data):
        
        batch_size = data.shape[0]
        data_hist = data[:,:,:,:-1]
        data_lr   = data[:,:,:,-1].reshape(batch_size, *self.shape_data[:-1])
        
        reco_hist = self.net_hist(data_hist)
        reco_lr   = self.net_lr(data_lr)
        reco_lr   = reco_lr.reshape(batch_size, self.shape_data[0],
                                    self.shape_data[1] * self.shape_data[2], 1)
        
        return torch.cat([reco_hist, reco_lr], dim = -1)
    #end
#end


class FokkerPlankNet_(nn.Module):
    def __init__(self, shape_data, config_params):
        super(FokkerPlankNet_, self).__init__()
        
        batch_size, timesteps, height, width, hbins = shape_data
        nbins = config_params.WIND_BINS.__len__() - 1
        self.conv_1d = nn.Conv1d(timesteps, timesteps, kernel_size = 5, padding = 2)
        self.dense_ae = nn.Sequential(
            nn.Linear(nbins, 5), nn.LeakyReLU(0.1),
            nn.Linear(5, nbins), nn.Softmax(dim = -1)
        )
        self.conv_2d = nn.Conv2d(nbins, nbins, kernel_size = 3, padding = 1)
    #end
    
    def forward(self, x):
        # x.shape : (m, T, H, W, B)
        batch_size, timesteps, height, width, nbins = x.shape
        
        # Model computations
        ## Block 1: Conv1d on timesteps
        x_ = x.reshape(batch_size, timesteps, height * width * nbins)
        x_ = self.conv_1d(x_)
        
        ## Reshape
        x_ = x_.reshape(batch_size, timesteps, height, width, nbins)
        
        ## Block 2: Conv2d on spatial dimensions (number of bins treated as channels)
        ## Timesteps treated independently
        
        # x_ = x_.transpose(2,4)
        # for t in range(timesteps):
        #     x_[:,t,:,:,:] = self.conv_2d(x_[:,t,:,:,:])
        # #end
        # x_ = x_.transpose(2,4)
        
        ## Block 3: Dense on histogram bins
        x_ = x_.reshape(batch_size, timesteps, height, width, nbins)
        x_ = x_.reshape(batch_size, timesteps, height * width, nbins)
        x_ = self.dense_ae(x_)
        x_ = x_.reshape(batch_size, timesteps, height, width, nbins)
        
        # Return
        return x_
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

class ModelObs_MM_uv(ModelObs_MM_mod):
    '''
    Variational cost:
        U(...) = lr_wind + lr_costh + lr_sinth + 
                 hr_wind_spatial + hr_wind_situ + 
                 hr_theta_spatial
    '''
    def __init__(self, shape_data, buoys_positions, dim_obs):
        super(ModelObs_MM_uv, self).__init__(shape_data, buoys_positions, dim_obs)
        
        self.dim_obs = dim_obs
        in_channels = self.shape_data[1]
        
        self.net_state_angle = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
        )
        
        self.net_data_wind = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size = (5,5)),
            nn.AvgPool2d((7,7)),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size = (5,5)),
            nn.MaxPool2d((7,7)),
            nn.LeakyReLU(0.1),
        )
    #end
    
    def forward(self, x, y_obs, mask):
        
        # Part of wind modulus
        dy_modulus = ModelObs_MM_mod.forward(self, x[:,:72], y_obs[:,:72], mask[:,:72])
        
        # Inclusion of angle data/state
        ## costh_lr = x[:,72:96]
        ## senth_lr = x[:,144:168]
        ## costh_an = x[:,96:120]
        ## senth_an = x[:,168:192]
        
        if torch.all(x[:,168:192] == 0) and torch.all(x[:,96:120] == 0):
            x_theta_spatial = torch.atan2(x[:,144:168], x[:,72:96])
        else:
            x_theta_spatial = torch.atan2(x[:,144:168], x[:,72:96]) + torch.atan2(x[:,168:192], x[:,96:120])
        #end
        
        y_mwind_spatial = (y_obs[:,:24] + y_obs[:,24:48]).mul(mask[:,24:48])
        
        # || x_costh_lr - y_costh_lr ||² + || x_sinth_lr - y_sinth_lr ||²
        dy_costh_lr = (x[:,72:96] - y_obs[:,72:96]).mul(mask[:,:24])
        dy_sinth_lr = (x[:,144:168] - y_obs[:,144:168]).mul(mask[:,:24])
        
        # feature maps
        feat_theta = self.net_state_angle(x_theta_spatial)
        feat_mwind = self.net_data_wind(y_mwind_spatial)
        dy_theta_mwind = (feat_theta - feat_mwind) * 0.
        
        return [*dy_modulus, dy_theta_mwind, dy_costh_lr, dy_sinth_lr]
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

class ModelObs_MM2d_uv(ModelObs_MM2d):
    def __init__(self, shape_data, dim_obs):
        super(ModelObs_MM2d_uv, self).__init__(shape_data, dim_obs)
        
        self.net_state_angle = None
        self.net_data_angle = None
    #end
    
    def forward(self, x, y_obs, mask):
        pass
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
        
        self.net_data = nn.Sequentbatch_inputial(
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

class ModelObs_MM1d_uv(ModelObs_MM1d):
    '''
    Note: no angle term here. 
    Only in situ time series with no angle
    '''
    def __init__(self, shape_data, buoys_coords, dim_obs):
        super(ModelObs_MM1d_uv, self).__init__(shape_data, buoys_coords, dim_obs)
    #end
    
    def forward(self, x, y_obs, mask):
        pass
    #end
#end


###############################################################################
##### MODEL SELECTION #########################################################
###############################################################################

def model_selection(shape_data, config_params, components = False):
    
    if config_params.PRIOR == 'SN':
        if not components:
            return ConvNet(shape_data, config_params)
        else:
            return ConvNet_mwind_angle(shape_data, config_params)
        #end
    #end
    
    elif config_params.PRIOR == 'RN':
        return ResNet(shape_data, config_params)
    
    elif config_params.PRIOR == 'UN':
        return UNet(shape_data, config_params)
    
    elif config_params.PRIOR == 'UN1':
        return UNet1(shape_data[1] * 3, shape_data[1] * 3)
    
    elif config_params.PRIOR == 'UNpdf':
        return UNet_pdf(shape_data, config_params)
    
    elif config_params.PRIOR == 'UN4':
        return UNet4(shape_data[1] * 3, shape_data[1] * 3)
    
    elif config_params.PRIOR == 'MLP':
        return MLP(config_params, shape_data)
    
    elif config_params.PRIOR == 'FPN':
        return FokkerPlankNet(shape_data, config_params)
    
    else:
        raise NotImplementedError('No valid prior')
    #end
#end
