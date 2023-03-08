import sys
sys.path.append('../utls')

import torch
import pytorch_lightning as pl

import numpy as np
import datetime

from metrics import NormLoss
from dlmodels import ModelObs_MM, ModelObs_MM1d, ModelObs_MM2d, ModelObs_SM
import solver as NN_4DVar
import futls as fs

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    # torch.set_default_tensor_type(torch.cuda.DoubleTensor)
else:
    DEVICE = torch.device('cpu')
#end



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
        
        params = list()
        
        if self.hparams.inversion == 'gs':
            params.append(
                {'params'       : self.model.model_Grad.parameters(),
                  'lr'           : self.hparams.mgrad_lr,
                  'weight_decay' : self.hparams.mgrad_wd}
                )
            params.append(
                {'params'       : self.model.model_VarCost.parameters(),
                  'lr'           : self.hparams.varcost_lr,
                  'weight_decay' : self.hparams.varcost_wd}
            )
        #end
        
        if self.model.Phi.__class__ is list or self.model.Phi.__class__ is torch.nn.ModuleList:
            # for phi in self.model.Phi:
            params.append(
                {'params'       : self.model.Phi[0].parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd}
            )
            params.append(
                {'params'       : self.model.Phi[1].parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd}
            )
            params.append(
                {'params'       : self.model.Phi[2].parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd}
            )
            #end
        else:
            params.append(
                {'params'       : self.model.Phi.parameters(),
                  'lr'           : self.hparams.prior_lr,
                  'weight_decay' : self.hparams.prior_wd}    
            )
        #end
        
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
    # #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, out = self.forward(batch, batch_idx, phase = 'train')
        loss = metrics['loss']
        estimated_time = self.get_estimated_time()
        
        self.log('loss', loss,           on_step = True, on_epoch = True, prog_bar = True)
        self.log('time', estimated_time, on_step = False, on_epoch = True, prog_bar = True)
        
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
        
        # if self.hparams.wind_modulus:
        #     batch_size = batch.shape[0]
        # else:
        #     batch_size = batch[0].shape[0]
        # #end
        batch_size = batch[0].shape[0]
        
        self.save_test_loss(test_loss, batch_size)
        return metrics, outs
    #end
    
    def spatial_downsample_interpolate(self, data):
        
        pooled = fs.downsample_and_interpolate_spatially(data, self.hparams.lr_kernel_size)
        return pooled
    #end
    
    def get_osse_mask(self, data_shape):
        
        mask, mask_lr, mask_hr_dx1, mask_hr_dx2 = fs.get_data_mask(data_shape, 
                                                                   self.mask_land, 
                                                                   self.hparams.lr_mask_sfreq, 
                                                                   self.hparams.hr_mask_sfreq, 
                                                                   self.hparams.hr_mask_mode, 
                                                                   self.buoy_position, 
                                                                   self.hparams.mm_obsmodel)
        
        if True:
            # Assume that u(23h day0) = u(0h day1)
            mask_lr[:,-1,:,:] = 1.
        #end
        
        return mask, mask_lr, mask_hr_dx1, mask_hr_dx2
    #end
    
    def get_data_lr_delay(self, data_lr, timesteps = 24, timewindow_start = 6,
                          delay_max = 5, delay_min = -4):
        
        data_lr = fs.get_delay_bias(data_lr, self.hparams.lr_mask_sfreq, timesteps, 
                                    timewindow_start, delay_max, delay_min)
        return data_lr
    #end
    
    def get_data_lr_alpha(self, data_lr, timesteps = 24, timewindow_start = 6,
                          intensity_min = 0.5, intensity_max = 1.5):
        
        data_lr = fs.get_remodulation_bias(data_lr, self.hparams.lr_mask_sfreq, timesteps, 
                                           timewindow_start, intensity_min, intensity_max)
        return data_lr
    #end
    
    def interpolate_channelwise(self, data_lr, timesteps = 24):
        
        data_interpolated = fs.interpolate_along_channels(data_lr, self.hparams.lr_mask_sfreq, timesteps)
        return data_interpolated
    #end
    
    def get_persistence(self, data, scale, longer_series, timesteps = 24, timewindow_start = 6):
        
        if scale == 'hr':
            frequency = self.hparams.hr_mask_sfreq
        elif scale == 'lr':
            frequency = self.hparams.lr_mask_sfreq
        #end
        
        if frequency.__class__ is list:
            in_freq = list()
            for i in range(frequency.__len__()):
                in_freq.append(frequency[i] + timewindow_start)
            #end
        else:
            in_freq = frequency
        #end
        
        persistence = fs.get_persistency_model(data, in_freq)
        return persistence
    #end
#end


class LitModel_OSSE1_WindModulus(LitModel_Base):
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None):
        super(LitModel_OSSE1_WindModulus, self).__init__(config_params)
        
        self.start_time = start_time
        
        # Mask for land/sea locations and buoys positions
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
    
    def get_baseline(self, data_lr):
        
        return self.interpolate_channelwise(data_lr)
    #end
    
    def prepare_batch(self, data, timewindow_start = 6, timewindow_end = 30, timesteps = 24):
        
        # data_hr = data.clone()
        data_hr_u, data_hr_v = data[0], data[1]
        data_hr_gt = (data_hr_u.pow(2) + data_hr_v.pow(2)).sqrt()
        
        if False:
            # Downsample to obtain ERA5-like data
            data_lr_gt = self.spatial_downsample_interpolate(data_hr_gt)
        else:
            # Modulus obtained as modulus of LR components
            data_lr_u = self.spatial_downsample_interpolate(data_hr_u)
            data_lr_v = self.spatial_downsample_interpolate(data_hr_v)
            data_lr_gt = (data_lr_u.pow(2) + data_lr_v.pow(2)).sqrt()
        #end
        
        # Bias: random phase shift or amplitude remodulation
        # Note: this is really one simuated data thing. With real data
        # there are no issues related to +/- Delta t
        # (thanks goodness real data are themselves biased)
        if self.hparams.lr_sampl_delay:
            data_lr_obs = self.get_data_lr_delay(data_lr_gt.clone(), timesteps, timewindow_start)
        elif self.hparams.lr_intensity:
            data_lr_obs = self.get_data_lr_alpha(data_lr_gt.clone(), timesteps, timewindow_start)
        else:
            data_lr_obs = data_lr_gt.clone()
        #end
        
        # Alternative : persistence models
        if False:
            data_lr_obs = data_lr_obs.clone()
            data_hr_obs = data_hr_gt.clone()
        else:
            data_lr_obs = self.get_persistence(data_lr_obs, 'lr', longer_series = True)
            data_hr_obs = self.get_persistence(data_hr_gt, 'hr', longer_series = True)
        #end
        
        # NOTE: is in-situ time series are actually measured, these positions
        # in the persistence model must be filled with in-situ time series
        if self.hparams.hr_mask_mode == 'buoys':
            xp_buoys, yp_buoys = self.buoy_position[:,0], self.buoy_position[:,1]
            data_hr_obs[:,:, xp_buoys, yp_buoys] = data_hr_gt[:,:, xp_buoys, yp_buoys]
        #end
        
        # Obtain anomaly
        if self.hparams.hr_mask_sfreq is None:
            # Cases : baselines (interpolation of LR and super-resolution)
            # that is, when no HR data is observed whatsoever
            data_an_obs = data_lr_obs
        else:
            # All the other cases
            data_an_obs = (data_hr_obs - data_lr_obs)
        #end
        
        # Isolate the 24 central timesteps
        data_lr_gt  = data_lr_gt[:, timewindow_start : timewindow_end, :,:]
        data_lr_obs = data_lr_obs[:, timewindow_start : timewindow_end, :,:]
        data_hr_gt  = data_hr_gt[:, timewindow_start : timewindow_end, :,:]
        data_an_obs = data_an_obs[:, timewindow_start : timewindow_end, :,:]
        data_hr_obs = data_hr_obs[:, timewindow_start : timewindow_end, :,:]
        
        if True:
            # This modification makes persistence and naive initializations to match
            # That is, we assume to "close" each time series with u(23h, day0) = u(0h, day1)
            # Activate it to implemen this (then initializations with persistence and naive match)
            data_lr_obs[:,-1,:,:] = data_lr_gt[:,-1,:,:]
        #end
        
        # Temporal interpolation
        # data_lr_obs = self.interpolate_channelwise(data_lr_obs, timesteps)
        
        return data_lr_gt, data_lr_obs, data_hr_gt, data_an_obs, data_hr_obs
    #end
    
    def get_input_data_state(self, data_lr, data_an, data_hr, init_state = None):
        
        # Prepare observations
        input_data = torch.cat([data_lr, data_an, data_an], dim = 1)
        
        # Prepare state variable
        if init_state is not None:
            input_state = init_state
        else:
            input_state = torch.cat([data_lr, data_an, data_an], dim = 1)
        #end
        
        return input_data, input_state
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        # Get and manipulate the data as desidered
        data_lr, data_lr_obs, data_hr, data_an, data_hr_obs = self.prepare_batch(data)
        input_data, input_state = self.get_input_data_state(data_lr_obs, data_an, data_hr_obs, init_state)
        
        # Mask data
        mask, mask_lr, mask_hr_dx1,_ = self.get_osse_mask(data_hr.shape)
        
        input_state = input_state * mask
        input_data  = input_data * mask
        
        # Inverse problem solution
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            
            if self.hparams.inversion == 'fp':
                
                outputs = self.model.Phi(input_data)
                reco_lr = self.get_baseline(data_lr_obs.mul(mask_lr))
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'gs':
                
                # mask_4DVarNet = [mask_lr, mask_hr_dx1, mask]
                
                outputs, _,_,_ = self.model(input_state, input_data, mask)
                # reco_lr = outputs[:,:24,:,:]
                reco_lr = self.get_baseline(data_lr_obs.mul(mask_lr))
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'bl':
                
                outputs = self.model.Phi(input_data)
                # reco_lr = data_lr_obs.clone()
                reco_lr = self.get_baseline(data_lr_obs.mul(mask_lr))
                reco_hr = reco_lr + torch.mul(outputs[:,48:,:,:], 0.)
            #end
        #end
        
        if False:
            torch.save(data_lr, './diagnostics/wmod/data_lr_gt.pkl')
            torch.save(data_lr_obs, './diagnostics/wmod/data_lr_obs.pkl')
            torch.save(data_lr_obs.mul(mask_lr), './diagnostics/wmod/data_lr_obs_mask.pkl')
            torch.save(reco_lr, './diagnostics/wmod/reco_lr.pkl')
            torch.save(reco_hr, './diagnostics/wmod/reco_hr.pkl')
            torch.save(data_hr, './diagnostics/wmod/data_hr_gt.pkl')
            torch.save(data_an, './diagnostics/wmod/data_an.pkl')
            torch.save(data_hr_obs, './diagnostics/wmod/data_hr_obs.pkl')
        #end
        
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
        loss_grad_x = self.loss_fn((grad_data[1] - grad_reco[1]), mask = None)
        loss_grad_y = self.loss_fn((grad_data[0] - grad_reco[0]), mask = None)
        loss += (loss_grad_x + loss_grad_y) * self.hparams.grad_coeff
        
        ## Regularization
        if not self.hparams.inversion == 'bl':
            
            regularization = self.loss_fn( (outputs - self.model.Phi(outputs)), mask = None )
            loss += regularization * self.hparams.reg_coeff
        #end
        
        if loss.isnan():
            raise ValueError('Loss is nan')
        #end
        
        return dict({'loss' : loss}), outputs
    #end
#end


class LitModel_OSSE1_WindComponents(LitModel_Base):
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None):
        super(LitModel_OSSE1_WindComponents, self).__init__(config_params)
        
        self.start_time = start_time
        
        # Mask for land/sea locations and buoys positions
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
    
    def get_baseline_components(self, data_lr):
        pass
    #end
    
    def split_components(self, data):
        return data[:,:,:, :self.shape_data[-1]], data[:,:,:, -self.shape_data[-1]:]
    #end
    
    def concat_components(self, data_u, data_v, dim_cat = -1):
        return torch.cat([data_u, data_v], dim = dim_cat)
    #end
    
    def get_modulus(self, data_u, data_v):
        return (data_u.pow(2) + data_v.pow(2)).sqrt()
    #end
    
    def get_angle(self, w_mod, comp_u, comp_v, cos_sin = False):
        
        cos_theta = comp_v / w_mod
        sin_theta = comp_u / w_mod
        
        if torch.any(cos_theta > 1.) or torch.any(cos_theta < -1.):
            raise ValueError('COS > 1 or < -1')
        #end
        if torch.any(sin_theta > 1.) or torch.any(sin_theta < -1.):
            raise ValueError('SIN > 1 or < -1')
        #end
        
        if cos_sin:
            return cos_theta, sin_theta
        else:
            theta = torch.atan2(sin_theta, cos_theta)
            return theta
        #end
    #end
    
    def prepare_batch(self, batch, timewindow_start = 6, timewindow_end = 30, timesteps = 24):
        
        # Import the components
        mwind_hr_gt_u, mwind_hr_gt_v = batch[0], batch[1]
        
        # High reso ground truths
        mwind_hr_gt = self.get_modulus(mwind_hr_gt_u, mwind_hr_gt_v)
        theta_hr_gt = self.get_angle(mwind_hr_gt, mwind_hr_gt_u, mwind_hr_gt_v)
        
        # Downsample to obtain low-resolution
        mwind_lr_gt_u = self.spatial_downsample_interpolate(mwind_hr_gt_u)
        mwind_lr_gt_v = self.spatial_downsample_interpolate(mwind_hr_gt_v)
        
        # Low reso ground truths
        mwind_lr_gt = self.get_modulus(mwind_lr_gt_u, mwind_lr_gt_v)
        theta_lr_gt = self.get_angle(mwind_lr_gt, mwind_lr_gt_u, mwind_lr_gt_v)
        
        # Delay or bias
        mwind_lr_gt_join = self.concat_components(mwind_lr_gt_u, mwind_lr_gt_v)
        if self.hparams.lr_sampl_delay:
            mwind_lr_obs = self.get_mwind_lr_delay(mwind_lr_gt_join.clone(), timesteps, timewindow_start)
        elif self.hparams.lr_intensity:
            mwind_lr_obs = self.get_mwind_lr_alpha(mwind_lr_gt_join.clone(), timesteps, timewindow_start)
        else:
            mwind_lr_obs = mwind_lr_gt_join.clone()
        #end
        
        # Low reso observations
        mwind_lr_obs_u, mwind_lr_obs_v = self.split_components(mwind_lr_obs)
        mwind_lr_obs_u = self.get_persistence(mwind_lr_obs_u, 'lr', longer_series = True)
        mwind_lr_obs_v = self.get_persistence(mwind_lr_obs_v, 'lr', longer_series = True)
        mwind_lr_obs = self.get_modulus(mwind_lr_obs_u, mwind_lr_obs_v)
        theta_lr_obs = self.get_angle(mwind_lr_obs, mwind_lr_obs_u, mwind_lr_obs_v)
        
        # High reso observations
        mwind_hr_obs = self.get_persistence(mwind_hr_gt, 'hr', longer_series = True)
        theta_hr_obs = self.get_persistence(theta_hr_gt, 'hr', longer_series = True)
        
        # NOTE: is in-situ time series are actually measured, these positions
        # in the persistence model must be filled with in-situ time series
        if self.hparams.hr_mask_mode == 'buoys':
            xp_buoys, yp_buoys = self.buoy_position[:,0], self.buoy_position[:,1]
            mwind_hr_obs[:,:, xp_buoys, yp_buoys] = mwind_hr_gt[:,:, xp_buoys, yp_buoys]
            theta_hr_obs[:,:, xp_buoys, yp_buoys] = theta_hr_gt[:,:, xp_buoys, yp_buoys]
        #end
        
        # Obtain the anomalies
        if self.hparams.hr_mask_sfreq is None:
            mwind_an_obs = mwind_lr_obs
            theta_an_obs = theta_lr_obs
        else:
            mwind_an_obs = (mwind_hr_obs - mwind_lr_obs)
            theta_an_obs = (theta_hr_obs - theta_lr_obs)
        #end
        
        # Isolate the 24 central timesteps
        mwind_lr_gt  = mwind_lr_gt[:, timewindow_start : timewindow_end, :,:]
        theta_lr_gt  = theta_lr_gt[:, timewindow_start : timewindow_end, :,:]
        mwind_hr_gt  = mwind_hr_gt[:, timewindow_start : timewindow_end, :,:]
        theta_hr_gt  = theta_hr_gt[:, timewindow_start : timewindow_end, :,:]
        mwind_lr_obs = mwind_lr_obs[:, timewindow_start : timewindow_end, :,:]
        theta_lr_obs = theta_lr_obs[:, timewindow_start : timewindow_end, :,:]
        mwind_an_obs = mwind_an_obs[:, timewindow_start : timewindow_end, :,:]
        theta_an_obs = theta_an_obs[:, timewindow_start : timewindow_end, :,:]
        mwind_hr_obs = mwind_hr_obs[:, timewindow_start : timewindow_end, :,:]
        theta_hr_obs = theta_hr_obs[:, timewindow_start : timewindow_end, :,:]
        
        if True:
            mwind_lr_obs[:,-1,:,:] = mwind_lr_gt[:,-1,:,:]
            theta_hr_obs[:,-1,:,:] = theta_lr_obs[:,-1,:,:]
        #end
        
        prepared_batch = {
            'mwind_lr_gt'  : mwind_lr_gt,
            'theta_lr_gt'  : theta_lr_gt,
            'mwind_hr_gt'  : mwind_hr_gt,
            'theta_hr_gt'  : theta_hr_gt,
            'mwind_lr_obs' : mwind_lr_obs,
            'theta_lr_obs' : theta_lr_obs,
            'mwind_an_obs' : mwind_an_obs,
            'theta_an_obs' : theta_an_obs,
            'mwind_hr_obs' : mwind_hr_obs,
            'theta_hr_obs' : theta_hr_obs
        }
        
        return prepared_batch
    #end
    
    def get_input_data_state(self, mwind_lr, mwind_an, costh_lr, sinth_lr, costh_an, sinth_an, init_state = None):
        
        # Prepare observations
        chunk_mwind = torch.cat([mwind_lr, mwind_an, mwind_an], dim = 1)
        chunk_costh = torch.cat([costh_lr, costh_an, costh_an], dim = 1)
        chunk_sinth = torch.cat([sinth_lr, sinth_an, sinth_an], dim = 1)
        input_data  = torch.stack([chunk_mwind, chunk_costh, chunk_sinth], dim = -1)
        
        # Prepare state variable
        if init_state is not None:
            input_state = init_state
        else:
            input_state = torch.stack([chunk_mwind, chunk_costh, chunk_sinth], dim = -1)
        #end
        
        return input_data, input_state
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        # Prepare batch
        prepared_batch = self.prepare_batch(data)
        
        # Elements of prepared batch
        data_mwind_lr_obs = prepared_batch['mwind_lr_obs']   # LR observation modulus
        data_theta_lr_obs = prepared_batch['theta_lr_obs']   # LR observation angle
        data_mwind_an_obs = prepared_batch['mwind_an_obs']   # Anomaly observation modulus
        data_theta_an_obs = prepared_batch['theta_an_obs']   # Anomaly observation angle
        data_mwind_lr_gt  = prepared_batch['mwind_lr_gt']    # LR ground truth modulus
        data_theta_lr_gt  = prepared_batch['theta_lr_gt']    # LR ground truth angle
        data_mwind_hr_gt  = prepared_batch['mwind_hr_gt']    # HR ground truth modulus
        data_theta_hr_gt  = prepared_batch['theta_hr_gt']    # HR ground truth angle
        
        data_costh_lr_obs = torch.cos(data_theta_lr_obs)
        data_sinth_lr_obs = torch.sin(data_theta_lr_obs)
        data_costh_an_obs = torch.cos(data_theta_an_obs)
        data_sinth_an_obs = torch.sin(data_theta_an_obs)
        data_costh_lr_gt  = torch.cos(data_theta_lr_gt)
        data_sinth_lr_gt  = torch.sin(data_theta_lr_gt)
        data_costh_hr_gt  = torch.cos(data_theta_hr_gt)
        data_sinth_hr_gt  = torch.sin(data_theta_hr_gt)
        
        # Prepare input observation and state
        input_data, input_state = self.get_input_data_state(data_mwind_lr_obs, data_mwind_an_obs,
                                                            data_costh_lr_obs, data_sinth_lr_obs,
                                                            data_costh_an_obs, data_sinth_an_obs)
        
        # Mask data
        mask, mask_lr, mask_hr_dx1, mask_hr_dx2 = self.get_osse_mask(data_mwind_lr_obs.shape)
        mask_global = torch.stack([mask, mask, mask], dim = -1)
        
        input_state = input_state * mask_global
        input_data  = input_data * mask_global
        
        # Inverse problem solution
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            
            if self.hparams.inversion == 'fp':
                
                reco_mwind = self.model.Phi[0](input_data[:,:,:,:,0])
                reco_costh = self.model.Phi[1](input_data[:,:,:,:,1])
                reco_sinth = self.model.Phi[2](input_data[:,:,:,:,2])
                
                reco_mwind_lr = self.interpolate_channelwise(data_mwind_lr_obs.mul(mask_lr))
                reco_costh_lr = self.interpolate_channelwise(data_costh_lr_obs.mul(mask_lr))
                reco_sinth_lr = self.interpolate_channelwise(data_sinth_lr_obs.mul(mask_lr))
                reco_mwind_an = reco_mwind[:,48:,:,:]
                reco_costh_an = reco_costh[:,48:,:,:]
                reco_sinth_an = reco_sinth[:,48:,:,:]
                reco_theta_lr = torch.atan2(reco_sinth_lr, reco_costh_lr)
                reco_theta_an = torch.atan2(reco_sinth_an, reco_costh_an)
                reco_mwind_hr = reco_mwind_lr + reco_mwind_an * self.hparams.anomaly_coeff
                reco_theta_hr = reco_theta_lr + reco_theta_an * self.hparams.anomaly_coeff
                
            elif self.hparams.inversion == 'gs':
                
                outputs, _,_,_ = self.model(input_state, input_data, mask_global)
                reco_mwind = outputs[:,:,:,:,0]
                reco_costh = outputs[:,:,:,:,1]
                reco_sinth = outputs[:,:,:,:,2]
                
                reco_mwind_lr = reco_mwind[:,:24,:,:]
                reco_costh_lr = reco_costh[:,:24,:,:]
                reco_sinth_lr = reco_sinth[:,:24,:,:]
                reco_mwind_an = reco_mwind[:,48:,:,:]
                reco_costh_an = reco_costh[:,48:,:,:]
                reco_sinth_an = reco_sinth[:,48:,:,:]
                reco_theta_lr = torch.atan2(reco_sinth_lr, reco_costh_lr)
                reco_theta_an = torch.atan2(reco_sinth_an, reco_costh_an)
                reco_mwind_hr = reco_mwind_lr + reco_mwind_an * self.hparams.anomaly_coeff
                reco_theta_hr = reco_theta_lr + reco_theta_an * self.hparams.anomaly_coeff
                
            elif self.hparams.inversion == 'bl':
                
                reco_mwind = self.model.Phi[0](input_data[:,:,:,:,0])
                reco_costh = self.model.Phi[1](input_data[:,:,:,:,1])
                reco_sinth = self.model.Phi[2](input_data[:,:,:,:,2])
                
                reco_mwind_lr = self.interpolate_channelwise(data_mwind_lr_obs.mul(mask_lr))
                reco_costh_lr = self.interpolate_channelwise(data_costh_lr_obs.mul(mask_lr))
                reco_sinth_lr = self.interpolate_channelwise(data_sinth_lr_obs.mul(mask_lr))
                reco_mwind_an = reco_mwind[:,48:,:,:]
                reco_costh_an = reco_costh[:,48:,:,:]
                reco_sinth_an = reco_sinth[:,48:,:,:]
                reco_theta_lr = torch.atan2(reco_sinth_lr, reco_costh_lr)
                reco_theta_an = torch.atan2(reco_sinth_an, reco_costh_an)
                reco_mwind_hr = reco_mwind_lr + torch.mul(reco_mwind_an, 0.)
                reco_theta_hr = reco_theta_lr + torch.mul(reco_theta_an, 0.)
            #end
        #end
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            
            data_hr = torch.cat([data_mwind_hr_gt, data_theta_hr_gt], dim = -1)
            reco_hr = torch.cat([reco_mwind_hr, reco_theta_hr], dim = -1)
            
            self.save_samples({'data' : data_hr.detach().cpu(),
                               'reco' : reco_hr.detach().cpu()})
            
            if self.hparams.inversion == 'gs':
                self.save_var_cost_values(self.model.var_cost_values)
            #end
        #end
        
        # Loss
        ## Reconstruction loss
        ## both mod and angle
        loss_mwind_lr = self.loss_fn((data_mwind_lr_gt - reco_mwind_lr))
        loss_mwind_hr = self.loss_fn((data_mwind_hr_gt - reco_mwind_hr))
        loss = self.hparams.weight_lres * loss_mwind_lr + self.hparams.weight_hres * loss_mwind_hr
        
        loss_costh_lr = self.loss_fn((data_costh_lr_gt - reco_costh_lr)) * 5
        loss_costh_hr = self.loss_fn((data_costh_hr_gt - torch.cos(reco_theta_hr))) * 10
        loss += ( loss_costh_lr + loss_costh_hr )
        
        loss_sinth_lr = self.loss_fn((data_sinth_lr_gt - reco_sinth_lr)) * 5
        loss_sinth_hr = self.loss_fn((data_sinth_hr_gt - torch.sin(reco_theta_hr))) * 10
        loss += ( loss_sinth_lr + loss_sinth_hr )
        
        loss_angle = self.loss_fn((reco_theta_hr - data_theta_hr_gt))
        loss += loss_angle
        
        ## Gradient loss
        grad_data_mwind = torch.gradient(data_mwind_hr_gt, dim = (3,2))
        grad_reco_mwind = torch.gradient(reco_mwind_hr,    dim = (3,2))
        
        loss_grad_x = self.loss_fn((grad_data_mwind[1] - grad_reco_mwind[1]))
        loss_grad_y = self.loss_fn((grad_data_mwind[0] - grad_reco_mwind[0]))
        loss += (loss_grad_x + loss_grad_y) * self.hparams.grad_coeff
        
        # Regularization term
        if not self.hparams.inversion == 'bl':
            
            regularization  = self.loss_fn( (reco_mwind - self.model.Phi[0](reco_mwind)) )
            regularization += self.loss_fn( (reco_costh - self.model.Phi[1](reco_costh)) )
            regularization += self.loss_fn( (reco_sinth - self.model.Phi[2](reco_sinth)) )
            loss += regularization * self.hparams.reg_coeff
        #end
        
        return dict({'loss' : loss}), torch.stack([reco_mwind, reco_costh, reco_sinth])
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