
import sys
sys.path.append('../utls')

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

import numpy as np
import datetime

from metrics import NormLoss
from dlmodels import ModelObs_MM, ModelObs_MM1d, ModelObs_MM2d, ModelObs_SM
import solver as NN_4DVar



class LitModel_Base(pl.LightningModule):
    
    def __init__(self, config_params):
        super(LitModel_Base, self).__init__()
        
        self.__train_losses      = np.zeros(config_params.EPOCHS)
        self.__val_losses        = np.zeros(config_params.EPOCHS)
        self.__test_losses       = list()
        self.__test_batches_size = list()
        self.__samples_to_save   = list()
        self.__var_cost_values   = list()        
        
        # cparams
        # Hyper-parameters, learning and workflow
        self.hparams.batch_size             = config_params.BATCH_SIZE
        self.hparams.lr_kernel_size         = config_params.LR_KERNELSIZE
        self.hparams.inversion              = config_params.INVERSION
        self.hparams.hr_mask_mode           = config_params.HR_MASK_MODE
        self.hparams.hr_mask_sfreq          = config_params.HR_MASK_SFREQ
        self.hparams.lr_mask_sfreq          = config_params.LR_MASK_SFREQ
        self.hparams.mm_obsmodel            = config_params.MM_OBSMODEL
        self.hparams.lr_sampl_delay         = config_params.LR_SAMP_DELAY
        self.hparams.lr_intensity           = config_params.LR_INTENSITY
        self.hparams.patch_extent           = config_params.PATCH_EXTENT
        self.hparams.wind_modulus           = config_params.WIND_MODULUS
        self.hparams.anomaly_coeff          = config_params.ANOMALY_COEFF
        self.hparams.reg_coeff              = config_params.REG_COEFF
        self.hparams.grad_coeff             = config_params.GRAD_COEFF
        self.hparams.weight_hres            = config_params.WEIGHT_HRES
        self.hparams.weight_lres            = config_params.WEIGHT_LRES
        self.hparams.mod_h_lr               = config_params.MODEL_H_LR
        self.hparams.mod_h_wd               = config_params.MODEL_H_WD
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
        self.train_epochs                   = config_params.EPOCHS
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
    
    def save_var_cost_values(self, var_cost_values):
        
        self.__var_cost_values = var_cost_values
    #end
    
    def get_var_cost_values(self):
        
        return self.__var_cost_values
    #end
    
    def get_estimated_time(self):
        
        start_time = self.start_time
        time_now = datetime.datetime.now()
        elapsed_time = (time_now - start_time).seconds / 60
        est_time = elapsed_time * (self.train_epochs / (self.current_epoch + 1) - 1)
        
        return est_time
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
    
    def configure_optimizers(self):
        
        params = [
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
        if self.hparams.mm_obsmodel:
            print('Multi-modal obs model')
            params.append(
                {'params'       : self.model.model_H.parameters(),
                 'lr'           : self.hparams.mod_h_lr,
                 'weight_decay' : self.hparams.mod_h_wd}
            )
        else:
            print('Single-modal obs model')
        #end
        
        optimizers = torch.optim.Adam(params)
        return optimizers
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, out = self.forward(batch, batch_idx, phase = 'train')
        loss = metrics['loss']
        estimated_time = self.get_estimated_time()
        
        self.log('loss', loss,                          on_step = True, on_epoch = True, prog_bar = True)
        self.log('time', estimated_time,                on_step = False, on_epoch = True, prog_bar = True)
        # self.log('data_mean',  metrics['data_mean'],    on_step = True, on_epoch = True, prog_bar = False)
        # self.log('state_mean', metrics['state_mean'],   on_step = True, on_epoch = True, prog_bar = False)
        # self.log('params',     metrics['model_params'], on_step = True, on_epoch = True, prog_bar = False)
        # self.log('reco_mean',  metrics['reco_mean'],    on_step = True, on_epoch = True, prog_bar = False)
        # self.log('grad_reco',  metrics['grad_reco'],    on_step = True, on_epoch = True, prog_bar = False)
        # self.log('grad_data',  metrics['grad_data'],    on_step = True, on_epoch = True, prog_bar = False)
        # self.log('reg_loss',   metrics['reg_loss'],     on_step = True, on_epoch = True, prog_bar = False)
        
        return loss
    #end
    
    def training_epoch_end(self, outputs):
        
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.save_epoch_loss(loss, self.current_epoch, 'train')
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
        
        if self.hparams.wind_modulus:
            batch_size = batch.shape[0]
        else:
            batch_size = batch[0].shape[0]
        #end
        
        self.save_test_loss(test_loss, batch_size)
        return metrics, outs
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
                    if wfreq == 'lr':
                        mask[:, freq, :,:] = 1.
                    elif wfreq == 'hr':
                        mask_land = self.mask_land.to(mask.device)
                        mask[:, freq, :,:] = mask_land #self.mask_land
                    #end
                    
                elif freq.__class__ is int:
                    for t in range(mask.shape[1]):
                        if t % freq == 0:
                            if wfreq == 'lr':
                                mask[:,t,:,:] = 1.
                            elif wfreq == 'hr':
                                mask_land = self.mask_land.to(mask.device)
                                mask[:,t,:,:] = mask_land #self.mask_land
                            #end
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
        # ONLY IF THERE IS NO TRAINABLE OBS MODEL! 
        # Because if there is, the time series of in-situ observations, 
        # are isolated thanks to the buoys positions
        # High-reso dx2 : all zeroes
        if not self.hparams.mm_obsmodel:
            mask_hr_dx1 = self.get_HR_obspoints_mask(data_shape, mode = hr_obs_point)
        else:
            mask_hr_dx1 = torch.zeros(data_shape)
        #end
        mask_hr_dx2 = torch.zeros(data_shape)
        
        mask_lr = get_resolution_mask(lr_sfreq, mask_lr, 'lr')
        mask_hr_dx1 = get_resolution_mask(hr_sfreq, mask_hr_dx1, 'hr')
        
        mask = torch.cat([mask_lr, mask_hr_dx1, mask_hr_dx2], dim = 1)
        return mask, mask_lr, mask_hr_dx1, mask_hr_dx2
    #end
    
    def get_data_lr_delay(self, data_lr, timesteps = 24, timewindow_start = 6,
                          delay_max = 5, delay_min = -4):
        
        batch_size       = data_lr.shape[0]
        timesteps        = timesteps
        timewindow_start = timewindow_start
        delay_max        = delay_max
        delay_min        = delay_min
        
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
                if t_true % self.hparams.lr_mask_sfreq == 0:
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
    
    def get_data_lr_alpha(self, data_lr, timesteps = 24, timewindow_start = 6,
                          intensity_min = 0.5, intensity_max = 1.5):
        
        batch_size = data_lr.shape[0]
        
        for m in range(batch_size):
            intensity = np.random.uniform(intensity_min, intensity_max)
            for t in range(timesteps):
                t_true = t + timewindow_start
                if t_true % self.hparams.lr_mask_sfreq == 0:
                    try:
                        data_lr[m,t_true,:,:] = data_lr[:,t_true,:,:] * intensity
                    except:
                        pass
                    #end
                #end
            #end
        #end
        
        return data_lr
    #end
    
    def get_baseline(self, data_lr_, timesteps = 24):
        
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
#end


class LitModel_OSSE1_WindModulus(LitModel_Base):
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None):
        super(LitModel_OSSE1_WindModulus, self).__init__(config_params)
        
        self.start_time = start_time
        
        # Dynamical prior and mask for land/sea locations
        # self.Phi = Phi
        self.mask_land = torch.Tensor(land_buoy_coordinates[0])
        self.buoy_position = land_buoy_coordinates[1]
        # self.wdatamodule = wdatamodule
        
        # Loss function — parameters optimization
        self.loss_fn = NormLoss()
        
        # Case-specific cparams
        self.run = run
        self.automatic_optimization = True
        self.has_any_nan = False
        
        # Initialize gradient solver (LSTM)
        batch_size, ts_length, height, width = shape_data
        mgrad_shapedata = [ts_length * 3, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        # Choice of observation model
        if self.hparams.hr_mask_mode == 'buoys' and self.hparams.hr_mask_sfreq is not None and self.hparams.mm_obsmodel:
            # Case time series plus obs HR, trainable obs term of 1d features
            observation_model = ModelObs_MM(shape_data, self.buoy_position, wind_modulus = True, dim_obs = 3)    
            
        elif self.hparams.hr_mask_mode == 'zeroes' and self.hparams.mm_obsmodel:
            # Case obs HR, trainable obs term of 2D features
            observation_model = ModelObs_MM2d(shape_data, wind_modulus = True, dim_obs = 2)
            
        elif self.hparams.hr_mask_mode == 'buoys' and self.hparams.mm_obsmodel:
            # Case only time series, trainable obs term for in-situ data
            observation_model = ModelObs_MM1d(shape_data, self.buoy_position, wind_modulus = True, dim_obs = 2)
        
        else:
            # Case default. No trainable obs term at all
            observation_model = ModelObs_SM(shape_data, wind_modulus = True, dim_obs = 1)
        #end
        
        # Instantiation of the gradient solver
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi,                                                            # Prior
            observation_model,                                              # Observation model
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
        
    def prepare_batch(self, data, timewindow_start = 6, timewindow_end = 30, timesteps = 24):
        
        data_hr = data.clone()
        
        # Downsample to obtain ERA5-like data
        data_lr = self.avgpool2d_keepsize(data_hr)
        
        # Obtain anomaly
        data_an = (data_hr - data_lr)
        
        # Delay mode
        if self.hparams.lr_sampl_delay:
            data_lr_input = self.get_data_lr_delay(data_lr.clone(), timesteps, timewindow_start)
        elif self.hparams.lr_intensity:
            data_lr_input = self.get_data_lr_alpha(data_lr.clone(), timesteps, timewindow_start)
        else:
            data_lr_input = data_lr.clone()
        #end
        
        # Isolate the 24 central timesteps
        data_lr_input = data_lr_input[:, timewindow_start : timewindow_end, :,:]
        data_hr       = data_hr[:, timewindow_start : timewindow_end, :,:]
        data_lr       = data_lr[:, timewindow_start : timewindow_end, :,:]
        data_an       = data_an[:, timewindow_start : timewindow_end, :,:]
        
        # Temporal interpolation
        data_lr_input = self.get_baseline(data_lr_input, timesteps)
        
        return data_hr, data_lr, data_lr_input, data_an
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
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
        mask, mask_lr, mask_hr_dx1,_ = self.get_osse_mask(data_hr.shape, 
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
                
                outputs = self.model.Phi(input_data)
                reco_lr = data_lr_input.clone()   # NOTE : forse qui data_lr_input ???
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'gs':
                
                mask_4DVarNet = [mask_lr, mask_hr_dx1, mask]
                
                outputs, _,_,_ = self.model(input_state, input_data, mask_4DVarNet)
                reco_lr = outputs[:,:24,:,:]
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'bl':
                
                outputs = self.model.Phi(input_data)
                reco_lr = data_lr_input.clone()
                reco_hr = reco_lr + 0. * outputs[:,48:,:,:]
            #end
        #end
        
        _log_reco_hr_mean = torch.mean(reco_hr)
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            self.save_samples({'data' : data_hr.detach().cpu(),
                               'reco' : reco_hr.detach().cpu()})
            
            if self.hparams.inversion == 'gs':
                self.save_var_cost_values(self.model.var_cost_values)
            #end
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
            
            regularization = self.loss_fn( (outputs - self.model.Phi(outputs)), mask = None )
            loss += regularization * self.hparams.reg_coeff
            
            _log_reg_loss = regularization
        else:
            _log_reg_loss = torch.Tensor([0.])
        #end
        
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


class LitModel_OSSE1_WindComponents(LitModel_Base):
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None):
        super(LitModel_OSSE1_WindComponents, self).__init__(config_params)
        
        self.start_time = start_time
        
        # Dynamical prior and mask for land/sea locations
        self.mask_land = torch.Tensor(land_buoy_coordinates[0])
        self.buoy_position = land_buoy_coordinates[1]
        
        # Loss function — parameters optimization
        self.loss_fn = NormLoss()
        
        # Case-specific cparams
        self.run = run
        self.automatic_optimization = True
        self.has_any_nan = False
        self.shape_data = shape_data
        
        # Initialize gradient solver (LSTM)
        batch_size, ts_length, height, width = shape_data
        mgrad_shapedata = [ts_length * 3, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        # Choice of observation model
        if self.hparams.hr_mask_mode == 'buoys' and self.hparams.hr_mask_sfreq is not None and self.hparams.mm_obsmodel:
            # Case time series plus obs HR, trainable obs term of 1d features
            observation_model = ModelObs_MM(shape_data, self.buoy_position, wind_modulus = False, dim_obs = 4)
            
        elif self.hparams.hr_mask_mode == 'zeroes' and self.hparams.mm_obsmodel:
            # Case obs HR, trainable obs term of 2D features
            observation_model = ModelObs_MM2d(shape_data, wind_modulus = False, dim_obs = 3)
            
        elif self.hparams.hr_mask_mode == 'buoys' and self.hparams.mm_obsmodel:
            # Case only time series, trainable obs term for in-situ data
            observation_model = ModelObs_MM1d(shape_data, self.buoy_position, wind_modulus = False, dim_obs = 3)
        
        else:
            # Case default. No trainable obs term at all
            observation_model = ModelObs_SM(shape_data, wind_modulus = False, dim_obs = 1)
        #end
        
        # Instantiation of the gradient solver
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            Phi,                                                            # Prior
            observation_model,                                              # Observation model
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
    
    def prepare_batch(self, batch, timewindow_start = 6, timewindow_end = 30, timesteps = 24):
        
        # Import the components
        data_hr_u, data_hr_v = batch[0], batch[1]
        
        # Downsample to obtain low-resolution
        data_lr_u = self.avgpool2d_keepsize(data_hr_u)
        data_lr_v = self.avgpool2d_keepsize(data_hr_v)
        
        # Obtain the wind speed modulus as high-resolution observation
        data_hr = torch.cat([data_hr_u, data_hr_v], dim = -1)
        
        # Obtain the anomalies
        data_an_u = (data_hr_u.pow(2) + data_hr_v.pow(2)).sqrt() - data_lr_u
        data_an_v = (data_hr_u.pow(2) + data_hr_v.pow(2)).sqrt() - data_lr_v
        
        # Concatenate the two components
        data_lr = torch.cat([data_lr_u, data_lr_v], dim = -1)
        data_an = torch.cat([data_an_u, data_an_v], dim = -1)
        
        # Delay mode
        if self.hparams.lr_sampl_delay:
            data_lr_input = self.get_data_lr_delay(data_lr.clone(), timesteps, timewindow_start)
        elif self.hparams.lr_intensity:
            data_lr_input = self.get_data_lr_alpha(data_lr.clone(), timesteps, timewindow_start)
        else:
            data_lr_input = data_lr.clone()
        #end
        
        # Isolate the 24 central timesteps
        data_lr_input = data_lr_input[:, timewindow_start : timewindow_end, :,:]
        data_hr       = data_hr[:, timewindow_start : timewindow_end, :,:]
        data_lr       = data_lr[:, timewindow_start : timewindow_end, :,:]
        data_an       = data_an[:, timewindow_start : timewindow_end, :,:]
        
        # Temporal interpolation
        data_lr_input_u = data_lr_input[:,:,:,:self.shape_data[-1]]
        data_lr_input_v = data_lr_input[:,:,:,-self.shape_data[-1]:]
        data_lr_input_u = self.get_baseline(data_lr_input_u, timesteps)
        data_lr_input_v = self.get_baseline(data_lr_input_v, timesteps)
        data_lr_input = torch.cat([data_lr_input_u, data_lr_input_v], dim = -1)
        
        return data_hr, data_lr, data_lr_input, data_an
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        # Prepare batch
        data_hr, data_lr, data_lr_input, data_an = self.prepare_batch(data)
        data_hr_u = data_hr[:,:,:, :self.shape_data[-1]]
        data_hr_v = data_hr[:,:,:, -self.shape_data[-1]:]
        data_lr_u = data_lr[:,:,:, :self.shape_data[-1]]
        data_lr_v = data_lr[:,:,:, -self.shape_data[-1]:]
        
        # Prepare low-resolution data
        input_data = torch.cat([data_lr_input, data_an, data_an], dim = 1)
        
        # Prepare input state initialized
        if init_state is None:
            input_state = torch.cat((data_lr_input, data_an, data_an), dim = 1)
        else:
            input_state = init_state
        #end
        
        # Mask data
        mask, mask_lr, mask_hr_dx1, mask_hr_dx2 = self.get_osse_mask(data_hr_u.shape, 
                                  self.hparams.lr_mask_sfreq, 
                                  self.hparams.hr_mask_sfreq, 
                                  self.hparams.hr_mask_mode)
        mask = torch.cat([mask, mask], dim = -1)
        
        input_state = input_state * mask
        input_data  = input_data * mask
        
        # Inverse problem solution
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            
            if self.hparams.inversion == 'fp':
                
                outputs = self.model.Phi(input_data)
                reco_lr = data_lr_input.clone()
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'gs':
                
                mask_4DVarNet = [mask_lr, mask_hr_dx1, mask]
                
                outputs, _,_,_ = self.model(input_state, input_data, mask_4DVarNet)
                reco_lr = outputs[:,:24,:,:]
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'bl':
                
                outputs = self.model.Phi(input_data)
                reco_lr = data_lr_input.clone()
                reco_hr = reco_lr + outputs[:,48:,:,:]
            #end
        #end
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            self.save_samples({'data' : data_hr.detach().cpu(),
                               'reco' : reco_hr.detach().cpu()})
            
            if self.hparams.inversion == 'gs':
                self.save_var_cost_values(self.model.var_cost_values)
            #end
        #end
        
        # Loss
        ## Split the output state in (u,v) components
        reco_hr_u = reco_hr[:,:,:, :self.shape_data[-1]]
        reco_hr_v = reco_hr[:,:,:, -self.shape_data[-1]:]
        reco_lr_u = reco_lr[:,:,:, :self.shape_data[-1]]
        reco_lr_v = reco_lr[:,:,:, -self.shape_data[-1]:]
        
        ## Reconstruction loss
        ## both mod and uv
        data_hr = torch.sqrt( torch.pow(data_hr_u, 2) + torch.pow(data_hr_v, 2) )
        data_lr = torch.sqrt( torch.pow(data_lr_u, 2) + torch.pow(data_lr_v, 2) )
        reco_hr = torch.sqrt( torch.pow(reco_hr_u, 2) + torch.pow(reco_hr_v, 2) )
        reco_lr = torch.sqrt( torch.pow(reco_lr_u, 2) + torch.pow(reco_lr_v, 2) )
        loss_hr_mod = self.loss_fn((data_hr - reco_hr), mask = None)
        loss_lr_mod = self.loss_fn((data_lr - reco_lr), mask = None)
        loss_hr = self.loss_fn((reco_hr_u - data_hr_u), mask = None) + \
                  self.loss_fn((reco_hr_v - data_hr_v), mask = None)
        loss_lr = self.loss_fn((reco_lr_u - data_lr_u), mask = None) + \
                  self.loss_fn((reco_lr_v - data_lr_v), mask = None)
        
        loss = self.hparams.weight_lres * loss_lr + self.hparams.weight_hres * loss_hr
        loss += (loss_lr_mod + loss_hr_mod)
        
        ## Loss on gradients
        grad_data_u = torch.gradient(data_hr_u, dim = (3,2))
        grad_reco_u = torch.gradient(reco_hr_u, dim = (3,2))
        grad_data_u = torch.sqrt(grad_data_u[0].pow(2) + grad_data_u[1].pow(2))
        grad_reco_u = torch.sqrt(grad_reco_u[0].pow(2) + grad_reco_u[1].pow(2))
        grad_data_v = torch.gradient(data_hr_v, dim = (3,2))
        grad_reco_v = torch.gradient(reco_hr_v, dim = (3,2))
        grad_data_v = torch.sqrt(grad_data_v[0].pow(2) + grad_data_v[1].pow(2))
        grad_reco_v = torch.sqrt(grad_reco_v[0].pow(2) + grad_reco_v[1].pow(2))
        
        loss_grad_u = self.loss_fn((grad_data_u - grad_reco_u), mask = None)
        loss_grad_v = self.loss_fn((grad_data_v - grad_reco_v), mask = None)
        
        grad_data = torch.gradient(data_hr, dim = (3,2))
        grad_reco = torch.gradient(reco_hr, dim = (3,2))
        grad_data = torch.sqrt(grad_data[0].pow(2) + grad_data[1].pow(2))
        grad_reco = torch.sqrt(grad_reco[0].pow(2) + grad_reco[1].pow(2))
        
        ## both mod and uv
        loss += self.hparams.grad_coeff * (loss_grad_u + loss_grad_v)
        loss_grad_mod = self.loss_fn((grad_data - grad_reco), mask = None)
        loss += loss_grad_mod
        
        ## Regularization term
        if not self.hparams.inversion == 'bl':
            
            regularization = self.loss_fn( (outputs - self.model.Phi(outputs)), mask = None )
            loss += regularization * self.hparams.reg_coeff
        #end
        
        return dict({'loss' : loss}), outputs
    #end
#end


class LitModel_OSSE2_Distribution(LitModel_Base):
    
    def __init__(self, cparams):
        super(LitModel_OSSE2_Distribution, self).__init__(cparams)
        
    #end
    
    def compute_loss(self, batch, batch_idx, phase = 'train'):
        
        pass
    #end
#end

