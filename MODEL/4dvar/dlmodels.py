import sys
sys.path.append('../utls')

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import solver as NN_4DVar
from metrics import NormLoss

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    DEVICE = torch.device('cpu')
#end


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()
    #end
    
    def forward(self, signal):
        print(signal.shape)
        return signal
    #end
#end

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

class Block(nn.Sequential):
    
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Block, self).__init__(
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
            Block(ts_length, 32, 5, 2),
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

def model_selection(shape_data, config_params):
    
    if config_params.PRIOR == 'SN':
        return ConvNet(shape_data, config_params)
    elif config_params.PRIOR == 'RN':
        return ResNet(shape_data, config_params)
    elif config_params.PRIOR == 'UN':
        return UNet(shape_data, config_params)
    else:
        raise NotImplementedError('No valid prior')
    #end
#end


class ObsModel_Mask(nn.Module):
    ''' Observation model '''
    
    def __init__(self, shape_data, dim_obs, mr_kernelsize = None):
        super(ObsModel_Mask, self).__init__()
        
        # NOTE : chennels == time series length
        self.shape_data = shape_data
        self.dim_obs = dim_obs
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
        self.mr_kernelsize = mr_kernelsize
    #end
        
    def forward(self, x, y_obs, mask):
        
        if self.dim_obs == 1:
            
            obs_term = (x - y_obs).mul(mask)
            return obs_term
            
        else:
            
            dy1 = (x - y_obs).mul(mask)
            
            yhr = y_obs[:,25:50,:,:]
            xhr = x[:,25:50,:,:]
            
            ymr = F.avg_pool2d(yhr, kernel_size = self.mr_kernelsize)
            ymr = F.interpolate(ymr, tuple(y_obs.shape[-2:]), mode = 'bicubic', align_corners = False)
            xmr = F.avg_pool2d(xhr, kernel_size = self.mr_kernelsize)
            xmr = F.interpolate(xmr, tuple(x.shape[-2:]), mode = 'bicubic', align_corners = False)
            
            dy2 = (xmr - ymr).mul(mask[:,25:50,:,:])
            
            return [dy1, dy2]
        #end
    #end
#end


class LitModel_Base(pl.LightningModule):
    
    def __init__(self, cparams):
        super(LitModel_Base, self).__init__()
        
        self.__train_losses      = np.zeros(cparams.EPOCHS)
        self.__val_losses        = np.zeros(cparams.EPOCHS)
        self.__test_losses       = list()
        self.__test_batches_size = list()
        self.__samples_to_save   = list()
    #end
    
    def has_nans(self):
        
        for param in self.model.parameters():
            if torch.any(param.isnan()):
                self.has_any_nan = True
            #end
        #end
        
        return self.has_any_nan
    #end
    
    def save_test_loss(self, test_loss, batch_size):
        
        self.__test_losses.append(test_loss)
        self.__test_batches_size.append(batch_size)
    #end
    
    def get_test_loss(self):
        
        losses = torch.Tensor(self.__test_losses)
        bsizes = torch.Tensor(self.__test_batches_size)
        weighted_average = torch.sum(torch.mul(losses, bsizes)).div(bsizes.sum())
        
        return weighted_average
    #end
    
    def save_samples(self, samples):
        
        self.__samples_to_save.append(samples)
    #end
    
    def get_saved_samples(self):
        
        return self.__samples_to_save
    #end
    
    def save_epoch_loss(self, loss, epoch, quantity):
        
        if quantity == 'train':
            self.__train_losses[epoch] = loss.item()
        elif quantity == 'val':
            self.__val_losses[epoch] = loss.item()
        #end
    #end
    
    def get_learning_curves(self):
        
        return self.__train_losses, self.__val_losses
    #end
    
    def remove_saved_outputs(self):
        
        del self.__samples_to_save
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, out = self.forward(batch, batch_idx, phase = 'train')
        loss = metrics['loss']
        self.log('loss', loss,                          on_step = True, on_epoch = True, prog_bar = True)
        self.log('data_mean',  metrics['data_mean'],    on_step = True, on_epoch = True, prog_bar = False)
        self.log('state_mean', metrics['state_mean'],   on_step = True, on_epoch = True, prog_bar = False)
        self.log('params',     metrics['model_params'], on_step = True, on_epoch = True, prog_bar = False)
        self.log('reco_mean',  metrics['reco_mean'],    on_step = True, on_epoch = True, prog_bar = False)
        self.log('grad_reco',  metrics['grad_reco'],    on_step = True, on_epoch = True, prog_bar = False)
        self.log('grad_data',  metrics['grad_data'],    on_step = True, on_epoch = True, prog_bar = False)
        self.log('reg_loss',   metrics['reg_loss'],     on_step = True, on_epoch = True, prog_bar = False)
        
        return loss
    #end
    
    def training_epoch_end(self, outputs):
        
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.save_epoch_loss(loss, self.current_epoch, 'train')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #end
    #end
    
    def validation_step(self, batch, batch_idx):
        
        metrics, out = self.forward(batch, batch_idx, phase = 'train')
        val_loss = metrics['loss']
        self.log('val_loss', val_loss)
        
        return val_loss
    #end
    
    def validation_epoch_end(self, outputs):
        
        loss = torch.stack([out for out in outputs]).mean()
        self.save_epoch_loss(loss, self.current_epoch, 'val')
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        #end
    #end
    
    def test_step(self, batch, batch_idx):
        
        with torch.no_grad():
            metrics, outs = self.forward(batch, batch_idx, phase = 'test')
            
            test_loss = metrics['loss']
            self.log('test_loss', test_loss.item())
        #end
        
        self.save_test_loss(test_loss, batch.shape[0])
        return metrics, outs
    #end
#end


class LitModel_OSSE1(LitModel_Base):
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, config_params, run):
        super(LitModel_OSSE1, self).__init__(config_params)
        
        # Dynamical prior and mask for land/sea locations
        self.Phi = Phi
        self.mask_land = torch.Tensor(land_buoy_coordinates[0])
        self.buoy_position = land_buoy_coordinates[1]
        
        # Loss function — parameters optimization
        self.loss_fn = NormLoss()
        
        # Hyper-parameters, learning and workflow
        self.hparams.lr_kernel_size         = config_params.LR_KERNELSIZE   # NOTE : 15 for 150x150 img and 31 for 324x324 img
        self.hparams.mr_kernel_size         = config_params.MR_KERNELSIZE
        self.hparams.inversion              = config_params.INVERSION
        self.hparams.hr_mask_mode           = config_params.HR_MASK_MODE
        self.hparams.hr_mask_sfreq          = config_params.HR_MASK_SFREQ
        self.hparams.lr_mask_sfreq          = config_params.LR_MASK_SFREQ
        self.hparams.lr_sampl_delay         = config_params.LR_SAMP_DELAY
        self.hparams.patch_extent           = config_params.PATCH_EXTENT
        self.hparams.anomaly_coeff          = config_params.ANOMALY_COEFF
        self.hparams.reg_coeff              = config_params.REG_COEFF
        self.hparams.grad_coeff             = config_params.GRAD_COEFF
        self.hparams.weight_hres            = config_params.WEIGHT_HRES
        self.hparams.weight_lres            = config_params.WEIGHT_LRES
        self.hparams.mgrad_lr               = config_params.SOLVER_LR
        self.hparams.mgrad_wd               = config_params.SOLVER_WD
        self.hparams.prior_lr               = config_params.PHI_LR
        self.hparams.prior_wd               = config_params.PHI_WD
        self.hparams.learn_varcost_params   = config_params.LEARN_VC_PARAMS
        self.hparams.varcost_lr             = config_params.VARCOST_LR
        self.hparams.varcost_wd             = config_params.VARCOST_WD
        self.hparams.dim_grad_solver        = config_params.DIM_LSTM
        self.hparams.dropout                = config_params.SOL_DROPOUT
        self.hparams.n_solver_iter          = config_params.NSOL_ITER
        self.hparams.n_fourdvar_iter        = config_params.N_4DV_ITER
        self.hparams.automatic_optimization = False
        self.has_any_nan                    = False
        self.run                            = run
        
        # Initialize gradient solver (LSTM)
        batch_size, ts_length, height, width = shape_data
        mgrad_shapedata = [ts_length * 3, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            self.Phi,                                                       # Prior
            ObsModel_Mask(shape_data, dim_obs = 1),                         # Observation model
            NN_4DVar.model_GradUpdateLSTM(                                  # Gradient solver
                mgrad_shapedata,                                              # m_Grad : Shape data
                False,                                                        # m_Grad : Periodic BCs
                self.hparams.dim_grad_solver,                                 # m_Grad : Dim LSTM
                self.hparams.dropout,                                         # m_Grad : Dropout
            ),
            NormLoss(),                                                     # Norm Observation
            NormLoss(),                                                     # Norm Prior
            model_shapedata,                                                # Shape data
            self.hparams.n_solver_iter,                                     # Solver iterations
            alphaObs = alpha_obs,                                           # alpha observations
            alphaReg = alpha_reg,                                           # alpha regularization
            varcost_learnable_params = self.hparams.learn_varcost_params    # learnable varcost params
        )
    #end
    
    def configure_optimizers(self):
        
        optimizers = torch.optim.Adam(
            [
                {'params'       : self.model.model_Grad.parameters(),
                 'lr'           : self.hparams.mgrad_lr,
                 'weight_decay' : self.hparams.mgrad_wd},
                {'params'       : self.model.Phi.parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd},
                {'params'       : self.model.model_VarCost.parameters(),
                 'lr'           : self.hparams.varcost_lr,
                 'weight_decay' : self.hparams.varcost_wd}
            ]
        )
        return optimizers
    #end
    
    def forward(self, batch, batch_idx, phase = 'train'):
        
        state_init = None
        for n in range(self.hparams.n_fourdvar_iter):
            
            loss, outs = self.compute_loss(batch, batch_idx, iteration = n, phase = phase, init_state = state_init)
            if not self.hparams.inversion == 'bl': # because baseline does not return tensor output
                state_init = outs.detach()
            #end
        #end
        
        return loss, outs
    #end
    
    def avgpool2d_keepsize(self, data):
        
        img_size = data.shape[-2:]
        pooled = F.avg_pool2d(data, kernel_size = self.hparams.lr_kernel_size)
        pooled  = F.interpolate(pooled, size = tuple(img_size), mode = 'bicubic', align_corners = False)
        
        if not data.shape == pooled.shape:
            raise ValueError('Original and Pooled_keepsize data shapes mismatch')
        #end
        
        return pooled
    #end
    
    def get_HR_obspoints_mask(self, data_shape, mode):
        
        if mode.__class__ is list:
            mode = mode[0]
        #end
        
        if mode == 'center':
            
            center_h, center_w = data_shape[-2] // 2, data_shape[-1] // 2
            mask = torch.zeros(data_shape)
            mask[:,:, center_h, center_w] = 1.
            
        elif mode == 'buoy' or mode == 'buoys':
            
            buoy_coords = self.buoy_position
            mask = torch.zeros(data_shape)
            mask[:,:, buoy_coords[:,0], buoy_coords[:,1]] = 1.
            
        elif mode == 'patch':
            
            delta_x = self.hparams.patch_extent
            center_h, center_w = data_shape[-2] // 2, data_shape[-1] // 2
            mask = torch.zeros(data_shape)
            mask[:,:, center_h - delta_x : center_h + delta_x, 
                 center_w - delta_x : center_w + delta_x] = 1.
            
        elif mode == 'zeroes':
            
            mask = torch.zeros(data_shape)
            
        elif mode == 'points':
            
            mask = torch.zeros(data_shape)
            points_x = np.random.randint(0, data_shape[-2], 10)
            points_y = np.random.randint(0, data_shape[-1], 10)
            mask[:,:,points_x, points_y] = 1.
        else:
            raise ValueError('Mask mode not impletemented.')
        #end
        
        return mask
    #end
    
    def get_osse_mask(self, data_shape, lr_sfreq, hr_sfreq, hr_obs_point):
        
        def get_resolution_mask(freq, mask, wfreq):
            
            if freq is None:
                return mask
            else:
                if freq.__class__ is list:
                    mask[:, freq, :,:] = 1.
                elif freq.__class__ is int:
                    for t in range(mask.shape[1]):
                        if t % freq == 0:
                            if wfreq == 'lr':
                                mask[:,t,:,:] = 1.
                            elif wfreq == 'hr':
                                mask[:,t,:,:] = self.mask_land
                        #end
                    #end
                #end
            #end
            
            return mask
        #end
        
        # Low-reso pseudo-observations
        mask_lr = torch.zeros(data_shape)
        
        # High-reso dx1 : get according to spatial sampling regime.
        # This gives time series of local observations in specified points
        # High-reso dx2 : all zeroes
        mask_hr_dx1 = self.get_HR_obspoints_mask(data_shape, mode = hr_obs_point)
        mask_hr_dx2 = torch.zeros(data_shape)
                
        mask_lr = get_resolution_mask(lr_sfreq, mask_lr, 'lr')
        mask_hr_dx1 = get_resolution_mask(hr_sfreq, mask_hr_dx1, 'hr')
        
        mask = torch.cat([mask_lr, mask_hr_dx1, mask_hr_dx2], dim = 1)
        return mask
    #end
    
    def get_data_lr_delay(self, data_lr, timesteps = 25, timewindow_start = 6,
                          delay_max = 5, delay_min = -4):
        
        batch_size       = data_lr.shape[0]
        timesteps        = timesteps
        timewindow_start = timewindow_start
        delay_max        = delay_max
        delay_min        = delay_min
        
        for m in range(batch_size):
            # constant delay for sample (for all timesteps)
            delay = np.random.randint(delay_min, delay_max)
            for t in range(timesteps):
                t_true = t + timewindow_start
                
                # NON È CHE È t_true QUI ????
                
                if t % self.hparams.lr_mask_sfreq == 0:
                    try:
                        data_lr[m,t_true,:,:] = data_lr[m,t_true + delay, :,:]
                    except:
                        pass
                    #end
                #end
            #end
        #end
        
        return data_lr
    #end
    
    def get_baseline(self, data_lr_, timesteps = 25):
        
        img_shape = data_lr_.shape[-2:]
        timesteps = timesteps
        lr_sfreq  = self.hparams.lr_mask_sfreq
        
        # Isolate timesteps related to LR data
        data_lr = torch.zeros((data_lr_.shape[0], timesteps // lr_sfreq + 1, *img_shape))
        for t in range(timesteps):
            if t % lr_sfreq == 0:
                data_lr[:, t // lr_sfreq, :,:] = torch.Tensor(data_lr_[:,t,:,:])
            #end
        #end
        data_lr[:,-1,:,:] = torch.Tensor(data_lr_[:,-1,:,:] )
        
        # Interpolate channel-wise (that is, timesteps)
        baseline = F.interpolate(data_lr.permute(0,2,3,1), 
                                 [img_shape[0], timesteps], 
                                 mode = 'bicubic', align_corners = False)
        baseline = baseline.permute(0,3,1,2)
        
        return baseline
    #end
    
    def prepare_batch(self, data, timewindow_start = 6, timewindow_end = 30, timesteps = 25):
        
        data_hr = data.clone()
        
        # Downsample to obtain ERA-like data
        data_lr = self.avgpool2d_keepsize(data_hr)
        
        # Obtain anomaly
        data_an = data_hr - data_lr
        
        # Delay mode
        if self.hparams.lr_sampl_delay:
            data_lr_input = self.get_data_lr_delay(data_lr.clone(), timesteps, timewindow_start)
        else:
            data_lr_input = data_lr.clone()
        #end
        
        # Alpha mode
        # if ...
        
        # Isolate the 24 central timesteps
        data_lr_input = data_lr_input[:, timewindow_start : timewindow_end + 1, :,:]
        data_hr       = data_hr[:, timewindow_start : timewindow_end + 1, :,:]
        data_lr       = data_lr[:, timewindow_start : timewindow_end + 1, :,:]
        data_an       = data_an[:, timewindow_start : timewindow_end + 1, :,:]
        
        # Temporal interpolation
        data_lr_input = self.get_baseline(data_lr_input, timesteps)
        
        return data_hr, data_lr, data_lr_input, data_an
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        # Get the optimizer
        opt = self.optimizers()
        opt.zero_grad()
        
        # Get and manipulate the data as desider
        data_hr, data_lr, data_lr_input, data_an = self.prepare_batch(data)
        input_data = torch.cat((data_lr_input, data_an, data_an), dim = 1)
        
        # Prepare input state initialized
        if init_state is None:
            input_state = torch.cat((data_lr_input, data_an, data_an), dim = 1)
        else:
            input_state = init_state
        #end
        
        # Mask data
        mask = self.get_osse_mask(data_hr.shape, 
                                  self.hparams.lr_mask_sfreq, 
                                  self.hparams.hr_mask_sfreq, 
                                  self.hparams.hr_mask_mode)
        
        input_state = input_state * mask
        input_data  = input_data * mask
        
        # LOGGING — hopefully to be removed soon
        _log_data_mean = torch.mean(input_data)
        _log_state_mean = torch.mean(input_state)
        _log_model_params = torch.mean(
            torch.Tensor([ param.mean() for param in self.parameters() ])
        )
        
        # Inverse problem solution
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            
            if self.hparams.inversion == 'fp':
                
                outputs = self.Phi(input_data)
                reco_lr = data_lr_input.clone()   # NOTE : forse qui data_lr_input ???
                reco_an = outputs[:,50:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'gs':
                
                outputs, _,_,_ = self.model(input_state, input_data, mask)
                reco_lr = outputs[:,:25,:,:]
                reco_an = outputs[:,50:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'bl':
                
                outputs = self.Phi(input_data)
                reco_lr = data_lr_input.clone()
                reco_hr = reco_lr + 0. * outputs[:,50:,:,:]
            #end
        #end
        
        _log_reco_hr_mean = torch.mean(reco_hr)
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            self.save_samples({'data' : data_hr.detach().cpu(), 
                               'reco' : reco_hr.detach().cpu()})
        #end
        
        # Compute loss
        ## Reconstruction loss
        loss_lr = self.loss_fn( (reco_lr - data_lr), mask = None )
        loss_hr = self.loss_fn( (reco_hr - data_hr), mask = None )
        loss = self.hparams.weight_lres * loss_lr + self.hparams.weight_hres * loss_hr
        
        ## Loss on gradients
        grad_data = torch.gradient(data_hr, dim = (3,2))
        grad_reco = torch.gradient(reco_hr, dim = (3,2))
        grad_data = torch.sqrt(grad_data[0].pow(2) + grad_data[1].pow(2))
        grad_reco = torch.sqrt(grad_reco[0].pow(2) + grad_reco[1].pow(2))
        
        # LOG GRADIENTS. Then I'll remove all this joke of variables, promised
        _log_grad_reco_loss = torch.mean(grad_reco)
        _log_grad_data_loss = torch.mean(grad_data)
        
        loss_grad = self.loss_fn((grad_data - grad_reco), mask = None)
        loss += loss_grad * self.hparams.grad_coeff
        
        ## Regularization
        if not self.hparams.inversion == 'bl':
            
            regularization = self.loss_fn( (outputs - self.Phi(outputs)), mask = None )
            loss += regularization * self.hparams.reg_coeff
            
            _log_reg_loss = regularization
        else:
            _log_reg_loss = torch.Tensor([0.])
        #end
        
        # Manual backward, to use mixed precision
        self.manual_backward(loss)
        opt.step()
        
        return dict({'loss' : loss,
                     'data_mean'    : _log_data_mean,
                     'state_mean'   : _log_state_mean,
                     'model_params' : _log_model_params,
                     'reco_mean'    : _log_reco_hr_mean,
                     'grad_reco'    : _log_grad_reco_loss,
                     'grad_data'    : _log_grad_data_loss,
                      'reg_loss'     : _log_reg_loss
                     }), outputs
    #end
#end


class LitModel_OSSE1_2(LitModel_Base):
    
    def __init__(self, cparams):
        super(LitModel_OSSE1_2, self).__init__(cparams)
        
    #end
    
    def compute_loss(self, batch, batch_idx, phase = 'train'):
        
        pass
    #end
#end