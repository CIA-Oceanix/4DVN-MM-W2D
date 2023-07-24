import sys
sys.path.append('../utls')

import torch
import pytorch_lightning as pl

import numpy as np
import datetime

from metrics import L2_Loss, L1_Loss, HellingerDistance, KLDivLoss
import dlmodels as dlm
import solver as NN_4DVar
import futls as fs

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
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
        self.hparams.weight_angle_hr        = config_params.WEIGHT_ANGLE_HR
        self.hparams.mod_h_lr               = config_params.MODEL_H_LR
        self.hparams.mod_h_wd               = config_params.MODEL_H_WD
        self.hparams.mgrad_lr               = config_params.SOLVER_LR
        self.hparams.mgrad_wd               = config_params.SOLVER_WD
        self.hparams.prior_lr               = config_params.PHI_LR
        self.hparams.prior_wd               = config_params.PHI_WD
        self.hparams.prior_cos_lr           = config_params.PHI_COS_LR
        self.hparams.prior_cos_wd           = config_params.PHI_COS_WD
        self.hparams.prior_sin_lr           = config_params.PHI_SIN_LR
        self.hparams.prior_sin_wd           = config_params.PHI_SIN_WD
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
            
            params.append(
                {'params'       : self.model.Phi[0].parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd}
            )
            params.append(
                {'params'       : self.model.Phi[1].parameters(),
                 'lr'           : self.hparams.prior_cos_lr,
                 'weight_decay' : self.hparams.prior_cos_wd}
            )
            params.append(
                {'params'       : self.model.Phi[2].parameters(),
                 'lr'           : self.hparams.prior_sin_lr,
                 'weight_decay' : self.hparams.prior_sin_wd}
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
    #end
    
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
        self.loss_fn = L2_Loss()
        
        # Case-specific cparams
        self.run = run
        self.automatic_optimization = True
        self.has_any_nan = False
        self.shape_data = shape_data
        
        # Initialize gradient solver (LSTM)
        # NOTE: this :4 is to do it compatible with LitModel_OSSE2_Distribution !!!
        batch_size, ts_length, height, width = shape_data[:4]
        mgrad_shapedata = [ts_length * 3, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        # Choice of observation model
        if self.hparams.hr_mask_mode == 'buoys' and self.hparams.hr_mask_sfreq is not None and self.hparams.mm_obsmodel:
            # Case time series plus obs HR, trainable obs term of 1d features
            observation_model = dlm.ModelObs_MM_mod(shape_data, self.buoy_position, dim_obs = 3)    
            
        elif self.hparams.hr_mask_mode == 'zeroes' and self.hparams.mm_obsmodel:
            # Case obs HR, trainable obs term of 2D features
            observation_model = dlm.ModelObs_MM2d_mod(shape_data, dim_obs = 2)
            
        elif self.hparams.hr_mask_mode == 'buoys' and self.hparams.mm_obsmodel:
            # Case only time series, trainable obs term for in-situ data
            observation_model = dlm.ModelObs_MM1d_mod(shape_data, self.buoy_position, dim_obs = 2)
        
        else:
            # Case default. No trainable obs term at all
            observation_model = dlm.ModelObs_SM(shape_data, dim_obs = 1)
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
            L2_Loss(),                                                      # Norm Observation
            L2_Loss(),                                                      # Norm Prior
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
        # These loops are ugly as fuck but necessary for the experiment on buoys shuffling
        if self.hparams.hr_mask_mode == 'buoys':
            xp_buoys, yp_buoys, logical_flag = self.buoy_position[:,0], self.buoy_position[:,1], self.buoy_position[:,2]
            for i in range(xp_buoys.shape[0]):
                if logical_flag[i] == 1:
                    data_hr_obs[:,:, xp_buoys[i], yp_buoys[i]] = data_hr_gt[:,:, xp_buoys[i], yp_buoys[i]]
                #end
            #end
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
                
                outputs, _,_,_ = self.model(input_state, input_data, mask)
                reco_lr = self.get_baseline(data_lr_obs.mul(mask_lr))
                reco_an = outputs[:,48:,:,:]
                reco_hr = reco_lr + self.hparams.anomaly_coeff * reco_an
                
            elif self.hparams.inversion == 'bl':
                
                outputs = self.model.Phi(input_data)
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
        self.l2_loss = L2_Loss()
        self.l1_loss = L1_Loss()
        
        # Case-specific cparams
        self.run = run
        self.automatic_optimization = True
        self.has_any_nan = False
        self.shape_data = shape_data
        
        # Initialize gradient solver (LSTM)
        batch_size, ts_length, height, width = shape_data
        mgrad_shapedata = [ts_length * 9, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        # TIP next modify dim_obs
        # Choice of observation model
        if self.hparams.hr_mask_mode == 'buoys' and self.hparams.hr_mask_sfreq is not None and self.hparams.mm_obsmodel:
            # Case time series plus obs HR, trainable obs term of 1d features
            observation_model = dlm.ModelObs_MM_uv(shape_data, self.buoy_position, dim_obs = 6)
            
        elif self.hparams.hr_mask_mode == 'zeroes' and self.hparams.mm_obsmodel:
            # Case obs HR, trainable obs term of 2D features
            observation_model = dlm.ModelObs_MM2d_uv(shape_data, dim_obs = 3)
            
        elif self.hparams.hr_mask_mode == 'buoys' and self.hparams.mm_obsmodel:
            # Case only time series, trainable obs term for in-situ data
            observation_model = dlm.ModelObs_MM1d_uv(shape_data, self.buoy_position, dim_obs = 3)
            
        else:
            # Case default. No trainable obs term at all
            observation_model = dlm.ModelObs_SM(shape_data, dim_obs = 1)
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
            L2_Loss(),                                                      # Norm Observation
            L2_Loss(),                                                      # Norm Prior
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
    
    def get_baseline(self, data_lr, apply_tanh = False):
        
        interpolated = self.interpolate_channelwise(data_lr)
        if apply_tanh:
            interpolated = torch.tanh(interpolated)
        #end
        
        return interpolated
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
        
        cos_theta = comp_u / w_mod
        sin_theta = comp_v / w_mod
        
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
        wind_hr_gt_u, wind_hr_gt_v = batch[0], batch[1]
        
        # High reso ground truths
        mwind_hr_gt = self.get_modulus(wind_hr_gt_u, wind_hr_gt_v)
        theta_hr_gt = self.get_angle(mwind_hr_gt, wind_hr_gt_u, wind_hr_gt_v)
        
        # Downsample to obtain low-resolution
        ## GT LR COMPONENTS
        wind_lr_gt_u = self.spatial_downsample_interpolate(wind_hr_gt_u)
        wind_lr_gt_v = self.spatial_downsample_interpolate(wind_hr_gt_v)
        
        # Low reso ground truths
        ## GT LR MODULUS AND ANGLE
        mwind_lr_gt = self.get_modulus(wind_lr_gt_u, wind_lr_gt_v)
        theta_lr_gt = self.get_angle(mwind_lr_gt, wind_lr_gt_u, wind_lr_gt_v)
        
        # Delay or bias
        ## THIS IS APPLIED TO GT LR COMPONENTS
        ### Join them so to bias consistently (it makes no sense to shift the two 
        ### components independently. In reality we have the NWP of (u,v) and if there
        ### is a bias in the prediction then it affects both components)
        wind_lr_gt_join = self.concat_components(wind_lr_gt_u, wind_lr_gt_v)
        
        ### Proper bias
        if self.hparams.lr_sampl_delay:
            wind_lr_obs = self.get_mwind_lr_delay(wind_lr_gt_join.clone(), timesteps, timewindow_start)
        elif self.hparams.lr_intensity:
            wind_lr_obs = self.get_mwind_lr_alpha(wind_lr_gt_join.clone(), timesteps, timewindow_start)
        else:
            wind_lr_obs = wind_lr_gt_join.clone()
        #end
        
        # Low reso observations
        ## Concatenate
        wind_lr_obs_u, wind_lr_obs_v = self.split_components(wind_lr_obs)
        
        ## Obtain persistences. NOTE: persistences affect those timesteps selected by
        ## our handy-candy list/integer specifying the sampling regimen. The timestep
        ## affected by bias IS THE SAME SELECTED FOR THE PERSISTENCE MODEL
        ## Like this: 
        ##      // Bias
        ##      data_lr[tt] = data_lr=[tt + Delta t]                  // data at timestep tt has swapped of Delta t (random delay)
        ##      // Persistence
        ##      data_lr[tt : tt + sampling_freq_rule] = data_lr[tt]   // persistence model takes the value previously modified
        ## That's it
        wind_lr_obs_u = self.get_persistence(wind_lr_obs_u, 'lr', longer_series = True)
        wind_lr_obs_v = self.get_persistence(wind_lr_obs_v, 'lr', longer_series = True)
        
        ## Get modulus and angle out of the persistenced models
        mwind_lr_obs  = self.get_modulus(wind_lr_obs_u, wind_lr_obs_v)
        theta_lr_obs  = self.get_angle(mwind_lr_obs, wind_lr_obs_u, wind_lr_obs_v)
        
        # High reso observations
        mwind_hr_obs = self.get_persistence(mwind_hr_gt, 'hr', longer_series = True)
        theta_hr_obs = self.get_persistence(theta_hr_gt, 'hr', longer_series = True)
        
        # NOTE: is in-situ time series are actually measured, these positions
        # in the persistence model must be filled with in-situ time series
        # NOTE 2: This makes sense, it is a DESIGN CHOICE. Since the data are multi-modal,
        # then this next step is a RAW MULTI-MODAL FUSION. The series are attached so 
        # to MATCH the physical locations of observations. This will come handy later on
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
        mwind_lr_gt   = mwind_lr_gt[:, timewindow_start : timewindow_end, :,:]
        theta_lr_gt   = theta_lr_gt[:, timewindow_start : timewindow_end, :,:]
        mwind_hr_gt   = mwind_hr_gt[:, timewindow_start : timewindow_end, :,:]
        theta_hr_gt   = theta_hr_gt[:, timewindow_start : timewindow_end, :,:]
        mwind_lr_obs  = mwind_lr_obs[:, timewindow_start : timewindow_end, :,:]
        theta_lr_obs  = theta_lr_obs[:, timewindow_start : timewindow_end, :,:]
        mwind_an_obs  = mwind_an_obs[:, timewindow_start : timewindow_end, :,:]
        theta_an_obs  = theta_an_obs[:, timewindow_start : timewindow_end, :,:]
        mwind_hr_obs  = mwind_hr_obs[:, timewindow_start : timewindow_end, :,:]
        theta_hr_obs  = theta_hr_obs[:, timewindow_start : timewindow_end, :,:]
        
        if True:
            mwind_lr_obs[:,-1,:,:]  = mwind_lr_gt[:,-1,:,:]
            wind_lr_obs_u[:,-1,:,:] = wind_lr_gt_u[:,-1,:,:]
            wind_lr_obs_v[:,-1,:,:] = wind_lr_gt_v[:,-1,:,:]
            theta_hr_obs[:,-1,:,:]  = theta_lr_obs[:,-1,:,:]
        #end
        
        prepared_batch = {
            'mwind_lr_gt'   : mwind_lr_gt,
            'theta_lr_gt'   : theta_lr_gt,
            'mwind_hr_gt'   : mwind_hr_gt,
            'theta_hr_gt'   : theta_hr_gt,
            'mwind_lr_obs'  : mwind_lr_obs,
            'theta_lr_obs'  : theta_lr_obs,
            'mwind_an_obs'  : mwind_an_obs,
            'theta_an_obs'  : theta_an_obs,
            'mwind_hr_obs'  : mwind_hr_obs,
            'theta_hr_obs'  : theta_hr_obs
        }
        
        return prepared_batch
    #end
    
    def get_input_data_state(self, mwind_lr, mwind_an, costh_lr, sinth_lr, init_state = None):
        
        # Prepare observations
        zeros_timeseries = torch.zeros(mwind_lr.shape)
        chunk_mwind = torch.cat([mwind_lr, mwind_an, mwind_an], dim = 1)
        chunk_costh = torch.cat([costh_lr, zeros_timeseries, zeros_timeseries], dim = 1)
        chunk_sinth = torch.cat([sinth_lr, zeros_timeseries, zeros_timeseries], dim = 1)
        input_data  = torch.cat([chunk_mwind, chunk_costh, chunk_sinth], dim = 1)
        
        # Prepare state variable
        if init_state is not None:
            input_state = init_state
        else:
            input_state = torch.cat([chunk_mwind, chunk_costh, chunk_sinth], dim = 1)
        #end
        
        return input_data, input_state
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        # Prepare batch
        prepared_batch = self.prepare_batch(data)
        
        # Elements of prepared batch
        data_mwind_lr_obs  = prepared_batch['mwind_lr_obs']   # LR observation modulus
        data_theta_lr_obs  = prepared_batch['theta_lr_obs']   # LR observation angle
        data_mwind_an_obs  = prepared_batch['mwind_an_obs']   # Anomaly observation modulus
        data_mwind_lr_gt   = prepared_batch['mwind_lr_gt']    # LR ground truth modulus
        data_theta_lr_gt   = prepared_batch['theta_lr_gt']    # LR ground truth angle
        data_mwind_hr_gt   = prepared_batch['mwind_hr_gt']    # HR ground truth modulus
        data_theta_hr_gt   = prepared_batch['theta_hr_gt']    # HR ground truth angle
        
        data_costh_lr_obs = torch.cos(data_theta_lr_obs)
        data_sinth_lr_obs = torch.sin(data_theta_lr_obs)
        data_costh_lr_gt  = torch.cos(data_theta_lr_gt)
        data_sinth_lr_gt  = torch.sin(data_theta_lr_gt)
        data_costh_hr_gt  = torch.cos(data_theta_hr_gt)
        data_sinth_hr_gt  = torch.sin(data_theta_hr_gt)
        
        # Prepare input observation and state
        input_data, input_state = self.get_input_data_state(data_mwind_lr_obs, data_mwind_an_obs,
                                                            data_costh_lr_obs, data_sinth_lr_obs)
        
        # Mask data
        mask, mask_lr, mask_hr_dx1, mask_hr_dx2 = self.get_osse_mask(data_mwind_lr_obs.shape)
        mask_ones   = torch.ones(mask_lr.shape)
        mask_zeros  = torch.zeros(mask_hr_dx1.shape)
        mask_fangl  = torch.cat([mask_ones, mask_zeros, mask_zeros], dim = 1)
        mask_global = torch.cat([mask, mask_fangl, mask_fangl], dim = 1)
        
        input_state = input_state * mask_global
        input_data  = input_data * mask_global
        
        # Inverse problem solution
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            
            if self.hparams.inversion == 'fp':
                
                outputs = self.model.Phi(input_data)
                reco_mwind = outputs[:, 0:72, :,:]
                reco_costh = outputs[:, 72:144, :,:]
                reco_sinth = outputs[:, 144:216, :,:]
                
                reco_mwind_lr = self.get_baseline(data_mwind_lr_obs.mul(mask_lr))
                reco_costh_lr = self.get_baseline(data_costh_lr_obs.mul(mask_lr), apply_tanh = True)
                reco_sinth_lr = self.get_baseline(data_sinth_lr_obs.mul(mask_lr), apply_tanh = True)
                
                reco_mwind_an = reco_mwind[:,48:,:,:]
                reco_costh_an = reco_costh[:,48:,:,:]
                reco_sinth_an = reco_sinth[:,48:,:,:]
                
                reco_theta_lr = torch.atan2(reco_sinth_lr, reco_costh_lr)
                reco_theta_an = torch.atan2(reco_sinth_an, reco_costh_an)
                reco_theta_hr = reco_theta_lr + reco_theta_an * self.hparams.anomaly_coeff
                reco_mwind_hr = reco_mwind_lr + reco_mwind_an * self.hparams.anomaly_coeff
                
            elif self.hparams.inversion == 'gs':
                
                outputs, _,_,_ = self.model(input_state, input_data, mask_global)
                reco_mwind = outputs[:, 0:72, :,:]
                reco_costh = outputs[:, 72:144, :,:]
                reco_sinth = outputs[:, 144:216, :,:]
                
                reco_mwind_lr = reco_mwind[:,:24,:,:]
                reco_costh_lr = reco_costh[:,:24,:,:]
                reco_sinth_lr = reco_sinth[:,:24,:,:]
                
                reco_mwind_an = reco_mwind[:,48:,:,:]
                reco_costh_an = reco_costh[:,48:,:,:]
                reco_sinth_an = reco_sinth[:,48:,:,:]
                
                reco_theta_lr = torch.atan2(reco_sinth_lr, reco_costh_lr)
                reco_theta_an = torch.atan2(reco_sinth_an, reco_costh_an)
                reco_theta_hr = reco_theta_lr + reco_theta_an * self.hparams.anomaly_coeff
                reco_mwind_hr = reco_mwind_lr + reco_mwind_an * self.hparams.anomaly_coeff
                
            elif self.hparams.inversion == 'bl':
                
                outputs = self.model.Phi(input_data)
                reco_mwind = outputs[:, 0:72, :,:]
                reco_costh = outputs[:, 72:144, :,:]
                reco_sinth = outputs[:, 144:216, :,:]
                
                reco_mwind_lr = self.get_baseline(data_mwind_lr_obs.mul(mask_lr))
                reco_costh_lr = self.get_baseline(data_costh_lr_obs.mul(mask_lr), apply_tanh = True)
                reco_sinth_lr = self.get_baseline(data_sinth_lr_obs.mul(mask_lr), apply_tanh = True)
                
                reco_mwind_an = reco_mwind[:,48:,:,:]
                reco_costh_an = reco_costh[:,48:,:,:]
                reco_sinth_an = reco_sinth[:,48:,:,:]
                
                reco_theta_lr = torch.atan2(reco_sinth_lr, reco_costh_lr)
                reco_theta_an = torch.atan2(reco_sinth_an, reco_costh_an)
                reco_mwind_hr = reco_mwind_lr + torch.mul(reco_mwind_an, 0.)
                reco_theta_hr = reco_theta_lr + torch.mul(reco_costh_an, 0.)
            #end
        #end
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            
            reco_hr = torch.cat([reco_mwind_hr, reco_theta_hr], dim = -1)
            data_hr = torch.cat([data_mwind_hr_gt, data_theta_hr_gt], dim = -1)
            
            self.save_samples({'data' : data_hr.detach().cpu(),
                               'reco' : reco_hr.detach().cpu()})
            
            if self.hparams.inversion == 'gs':
                self.save_var_cost_values(self.model.var_cost_values)
            #end
        #end
        
        # Loss
        ## Reconstruction loss
        ## both mod and angle
        loss_mwind_lr = self.l2_loss((data_mwind_lr_gt - reco_mwind_lr))
        loss_mwind_hr = self.l2_loss((data_mwind_hr_gt - reco_mwind_hr))
        loss = self.hparams.weight_lres * loss_mwind_lr + self.hparams.weight_hres * loss_mwind_hr
        
        loss_costh_lr = self.l1_loss((data_costh_lr_gt - reco_costh_lr))
        loss_costh_hr = self.l1_loss((data_costh_hr_gt - torch.cos(reco_theta_hr))) * self.hparams.weight_angle_hr
        loss += ( loss_costh_lr + loss_costh_hr )
        
        loss_sinth_lr = self.l1_loss((data_sinth_lr_gt - reco_sinth_lr))
        loss_sinth_hr = self.l1_loss((data_sinth_hr_gt - torch.sin(reco_theta_hr))) * self.hparams.weight_angle_hr
        loss += ( loss_sinth_lr + loss_sinth_hr )
        
        ''' Remove this part ??
        # Reconstruction on components
        # data_wind_u = data_mwind_hr_gt * torch.cos(data_theta_hr_gt)
        # data_wind_v = data_mwind_hr_gt * torch.sin(data_theta_hr_gt)
        # reco_wind_u = reco_mwind_hr * torch.cos(reco_theta_hr)
        # reco_wind_v = reco_mwind_hr * torch.sin(reco_theta_hr)
        # loss_u = self.l1_loss((data_wind_u - reco_wind_u))
        # loss_v = self.l1_loss((data_wind_v - reco_wind_v))
        # loss += ( loss_u + loss_v )
        '''
        
        ## Gradient loss
        grad_data_mwind = torch.gradient(data_mwind_hr_gt, dim = (3,2))
        grad_reco_mwind = torch.gradient(reco_mwind_hr,    dim = (3,2))
        
        loss_grad_x = self.l2_loss((grad_data_mwind[1] - grad_reco_mwind[1]))
        loss_grad_y = self.l2_loss((grad_data_mwind[0] - grad_reco_mwind[0]))
        loss += (loss_grad_x + loss_grad_y) * self.hparams.grad_coeff
        
        # Regularization term
        if not self.hparams.inversion == 'bl':
            
            regularization  = self.l2_loss( (outputs - self.model.Phi(outputs)) ) 
            loss += regularization * self.hparams.reg_coeff
        #end
        
        return dict({'loss' : loss}), torch.stack([reco_mwind, reco_costh, reco_sinth])
    #end
#end


class LitModel_OSSE2_Distribution(LitModel_OSSE1_WindModulus):
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None):
        super(LitModel_OSSE2_Distribution, self).__init__(Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None)
        
        self.start_time = start_time
        
        # Mask for land/sea locations and buoys positions
        self.mask_land = torch.Tensor(land_buoy_coordinates[0])
        self.buoy_position = land_buoy_coordinates[1]
        
        # Loss function — parameters optimization
        self.l2_loss = L2_Loss()
        self.l1_loss = L1_Loss()
        self.kl_loss = KLDivLoss()
        self.hd_loss = HellingerDistance()
        
        # Case-specific cparams
        self.wind_bins = config_params.WIND_BINS
        self.run = run
        self.automatic_optimization = True
        self.has_any_nan = False
        self.shape_data = shape_data
        self.wind_bins = config_params.WIND_BINS
        
        # Initialize gradient solver (LSTM)
        self.shape_data = shape_data
        batch_size, ts_length, height, width, hbins = shape_data
        mgrad_shapedata = [ts_length * 9, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        # Choice of observation model
        observation_model = observation_model = dlm.ModelObs_SM(shape_data, dim_obs = 1)
        
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
            L2_Loss(),                                                      # Norm Observation
            L2_Loss(),                                                      # Norm Prior
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
    
    def hist_to_img_owa(self, data, mode):
        
        batch_size, timesteps,height, width, hbins  = self.shape_data
        if mode == 'h2i':
            return data.reshape(batch_size, timesteps, height * width, hbins)
        elif mode == 'i2h':
            return data.reshape(batch_size, timesteps, height, width, hbins)
        else:
            raise ValueError('Reshape mode not implemented')
        #end
    #end
    
    def prepare_batch(self, batch, timewindow_start = 6, timewindow_end = 30, timesteps = 24):
        
        wind_hist, wind_lr = batch
        batch_size, timesteps, height, width, hbins = wind_hist.shape
        
        # Reshape to image
        wind_hist = wind_hist.reshape(batch_size, timesteps, height * width, hbins)
        
        # Persistences !!!
        data_hst_obs = self.get_persistence(wind_hist, 'hr', longer_series = True)
        data_lr_obs  = self.get_persistence(wind_lr,   'lr', longer_series = True)
        
        # Crop central timesteps
        data_hst_obs = data_hst_obs[:, timewindow_start : timewindow_end, :,:]
        data_lr_obs  = data_lr_obs[:, timewindow_start : timewindow_end, :,:]
        wind_hist    = wind_hist[:, timewindow_start : timewindow_end, :,:]
        
        # Reshape to histogram
        wind_hist    = wind_hist.reshape(batch_size, wind_hist.shape[1], height, width, hbins)
        data_hst_obs = data_hst_obs.reshape(batch_size, data_hst_obs.shape[1], height, width, hbins)
        
        return data_hst_obs, wind_hist, data_lr_obs, wind_lr
    #end
    
    def get_mask(self, shape_data):
        
        mask_lr   = fs.get_resolution_mask_(self.hparams.lr_mask_sfreq, torch.zeros(shape_data), self.mask_land, 'lr')
        mask_hist = fs.get_resolution_mask_(self.hparams.hr_mask_sfreq, torch.zeros(shape_data), self.mask_land, 'hr')
        mask_lr[0,-1,:,:] = 1.
        
        return mask_lr, mask_hist
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        wind_hist, wind_hist_gt, wind_lr, wind_lr_gt = self.prepare_batch(data)
        batch_size, timesteps, height, width = wind_lr_gt.shape
        
        # Mask data
        mask_lr, mask_hist = self.get_mask(wind_hist[:,:,:,:,0].shape)
        wind_hist = wind_hist * mask_hist.unsqueeze(-1)
        batch_input = wind_hist.clone()
        
        # Inversion
        with torch.set_grad_enabled(True):
            batch_input = torch.autograd.Variable(batch_input, requires_grad = True)
            outputs = self.model.Phi(batch_input)
        #end
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            self.save_samples({
                'data' : wind_hist_gt.detach().cpu(),
                'reco' : outputs.detach().cpu()
            })
        #end
        
        loss_mse = self.l2_loss((wind_hist_gt - outputs))
        loss_kld = self.kl_loss(wind_hist_gt, outputs)
        loss_hd  = self.hd_loss(wind_hist_gt, outputs)
        loss = loss_kld + loss_mse + loss_hd
        
        return dict({'loss' : loss}), outputs
    #end
#end