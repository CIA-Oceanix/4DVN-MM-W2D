import sys
sys.path.append('../utls')
sys.path.append('./utls')

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
    
    def get_train_times(self):
        return self.start_time, self.end_time
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
        
        params.append(
            {'params'       : self.model.Phi.parameters(),
              'lr'           : self.hparams.prior_lr,
              'weight_decay' : self.hparams.prior_wd}    
        )
        
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
    
    def on_train_end(self):
        self.end_time = datetime.datetime.now()
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
        if (self.hparams.hr_mask_mode == 'buoys' or self.hparams.hr_mask_mode == 'fbuoys') and self.hparams.hr_mask_sfreq is not None and self.hparams.mm_obsmodel:
            # Case time series plus obs HR, trainable obs term of 1d features
            observation_model = dlm.ModelObs_MM_mod(shape_data, self.buoy_position, dim_obs = 3)    
            
        elif self.hparams.hr_mask_mode == 'zeroes' and self.hparams.mm_obsmodel:
            # Case obs HR, trainable obs term of 2D features
            observation_model = dlm.ModelObs_MM2d_mod(shape_data, dim_obs = 2)
            
        elif (self.hparams.hr_mask_mode == 'buoys' or self.hparams.hr_mask_mode == 'fbuoys') and self.hparams.mm_obsmodel:
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
    
    def prepare_batch(self, data, timewindow_start = 6, timewindow_end = 30, timesteps = 24, phase = 'train'):
        
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
        if self.hparams.lr_sampl_delay and phase != 'test':
            data_lr_obs = self.get_data_lr_delay(data_lr_gt.clone(), timesteps, timewindow_start)
        elif self.hparams.lr_intensity and phase != 'test':
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
        # These loops are not nice but necessary for the experiment on buoys shuffling
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
        data_lr, data_lr_obs, data_hr, data_an, data_hr_obs = self.prepare_batch(data, phase = phase)
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


class LitModel_OSSE2_Distribution(LitModel_OSSE1_WindModulus):
    # Now this will change a bit
    
    def __init__(self, Phi, shape_data, land_buoy_coordinates, normparams, config_params, run, start_time = None):
        super(LitModel_OSSE2_Distribution, self).__init__(Phi, shape_data, land_buoy_coordinates, config_params, run, start_time = None)
        
        self.start_time = start_time
        
        # Mask for land/sea locations and buoys positions
        self.mask_land     = torch.Tensor(land_buoy_coordinates[0])
        self.buoy_position = land_buoy_coordinates[1]
        self.normparams    = normparams
        
        # Loss function — parameters optimization
        self.l2_loss = L2_Loss()
        self.hd_loss = HellingerDistance()
        
        # Case-specific cparams
        self.run                    = run
        self.automatic_optimization = True
        self.has_any_nan            = False
        self.shape_data             = shape_data
        self.wind_bins              = config_params.WIND_BINS
        self.pretrained_prior       = config_params.LOAD_PT_WEIGHTS
        self.hparams.prior_hist_lr  = config_params.PHI_HIST_LR
        self.hparams.prior_hist_wd  = config_params.PHI_HIST_WD
        self.__hd_metric            = np.zeros(config_params.EPOCHS)
        self.__train_losses         = np.zeros(config_params.EPOCHS)
        self.__val_losses           = np.zeros(config_params.EPOCHS)
        
        # Initialize gradient solver (LSTM)
        self.shape_data = shape_data
        batch_size, ts_length, height, width, hbins = shape_data
        mgrad_shapedata = [ts_length * 3, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        # Choice of observation model
        observation_model = observation_model = dlm.ModelObs_SM(shape_data, dim_obs = 1)
        
        # Neural histogram regressor
        self.h_Phi = dlm.HistogrammizationDirect(shape_data[1], 128, shape_data, 
                                                 config_params.LR_KERNELSIZE,
                                                 config_params.WIND_BINS)
        
        # Choice of the inversion scheme
        if self.hparams.inversion == 'fp':
            self.model = Phi
            
        elif self.hparams.inversion == 'gs':
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
                varcost_learnable_params = self.hparams.learn_varcost_params,   # learnable varcost params
            )
        #end
    #end
    
    def forward(self, batch, batch_idx, phase = 'train'):
        
        state_init = None
        for n in range(self.hparams.n_fourdvar_iter):
            
            loss, outs, hd = self.compute_loss(batch, batch_idx, iteration = n, phase = phase, init_state = state_init)
            if not self.hparams.inversion == 'bl': # because baseline does not return tensor output
                state_init = outs.detach()
            #end
        #end
        
        return loss, outs, hd
    #end
    
    def save_hd_metric(self, hd_metric_end_epoch, epoch):
        self.__hd_metric[epoch] = hd_metric_end_epoch
    #end
    
    def get_learning_curves(self):
        return self.__train_losses, self.__val_losses, self.__hd_metric
    #end
    
    def save_epoch_loss(self, loss, epoch, quantity):
        
        if quantity == 'train':
            self.__train_losses[epoch] = loss.item()
        elif quantity == 'val':
            self.__val_losses[epoch] = loss.item()
        #end
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, out, hdist = self.forward(batch, batch_idx, phase = 'train')
        loss = metrics['loss']
        estimated_time = self.get_estimated_time()
        
        self.log('loss',      loss,           on_step = True,  on_epoch = True, prog_bar = True)
        self.log('time_left', estimated_time, on_step = False, on_epoch = True, prog_bar = True)
        self.log('h_dist',    hdist,          on_step = False, on_epoch = True, prog_bar = True)
        
        # return loss
        return {'loss' : loss, 'hdist' : hdist}
    #end
    
    def training_epoch_end(self, outputs):
        
        loss  = torch.stack([out['loss'] for out in outputs]).mean()
        hdist = torch.stack([out['hdist'] for out in outputs]).mean()
        self.save_hd_metric(hdist, self.current_epoch)
        self.save_epoch_loss(loss, self.current_epoch, 'train')
    #end
    
    def validation_step(self, batch, batch_idx):
        
        metrics, out,_ = self.forward(batch, batch_idx, phase = 'train')
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
            metrics, outs, hd = self.forward(batch, batch_idx, phase = 'test')
            
            test_loss = metrics['loss']
            self.log('test_loss', test_loss.item())
            self.log('test_hd',   hd.item())
        #end
        
        batch_size = batch[0].shape[0]
        
        self.save_test_loss(test_loss, batch_size)
        return metrics, outs
    #end
    
    def configure_optimizers(self):
        
        # Gather model parameters
        params = list()
        
        if self.hparams.inversion == 'gs':
            
            # Gradient Solver and Variational Cost parameters
            # Used only for 4DVarNet gradient solver training
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
        
        # Parameters of trainable fields-to-histogram map
        # These parameters are considered as trainable either way
        params.append(
            {'params'       : self.model.Phi.Phi_fields_to_hist.parameters(),
             'lr'           : self.hparams.prior_hist_lr,
             'weight_decay' : self.hparams.prior_hist_wd}
        )
        
        # If UNet parameters are not loaded as pretrained model, 
        # append parameters of UNet to fit them as well
        # These parameters are optional
        # if self.pretrained_prior is None:
        if False:
            params.append(
                {'params'       : self.model.Phi.Phi_fields_hr.parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd}
            )
        #end
        
        if self.hparams.mm_obsmodel:
            
            # Parameters of the trainable Observation Operator
            print('Multi-modal obs model')
            params.append(
                {'params'        : self.model.model_H.parameters(),
                  'lr'           : self.hparams.mod_h_lr,
                  'weight_decay' : self.hparams.mod_h_wd}
            )
        else:
            print('Single-modal obs model')
        #end
        
        # Define optimizer
        optimizer = torch.optim.Adam(params)
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.25)
    
        # Return dictionary
        optimizer_scheduler_dict = {
            'optimizer'    : optimizer,
            'lr_scheduler' : {
                'scheduler' : scheduler,
                'monitor'   : 'val_loss'
            }
        }
        
        return optimizer_scheduler_dict
    #end
    
    def load_ckpt_from_statedict(self, Phi_statedict):
        
        Phi_dict = dict()
        for key in Phi_statedict.keys():
            if key.startswith('model.Phi.'):
                Phi_dict.update({key.replace('model.Phi.', '') : Phi_statedict[key]})
            #end
        #end
        self.model.Phi.Phi_fields_hr.load_state_dict(Phi_dict)
    #end
    
    def test_epoch_end(self, outputs):
        
        print('TEST END. Downsample mask')
        self.downsample_mask()
    #end
    
    def downsample_mask(self):
        
        mask_downsampled = self.get_downsampled_mask()
        self.mask_land = mask_downsampled
    #end
    
    def get_downsampled_mask(self, mask = None):
        
        if mask is None:
            mask_land = self.mask_land
            mask_downsampled = torch.nn.functional.avg_pool2d(mask_land.unsqueeze(0).unsqueeze(0), self.hparams.lr_kernel_size)
        else:
            mask_downsampled = torch.nn.functional.avg_pool2d(mask, self.hparams.lr_kernel_size)
        #end
        
        mask_downsampled[mask_downsampled <= 0.5] = 0
        mask_downsampled[mask_downsampled > 0.5]  = 1
        
        return mask_downsampled
    #end
    
    def interpolate_channelwise(self, data_lr, timesteps = 24):
        
        data_interpolated = fs.interpolate_along_channels(data_lr, self.hparams.lr_mask_sfreq, timesteps)
        return data_interpolated
    #end
    
    def prepare_batch(self, batch, timewindow_start = 6, timewindow_end = 30, timesteps = 24):
        
        wind_hist, (data_hr_u, data_hr_v) = batch
        data_hr_gt = (data_hr_u.pow(2) + data_hr_v.pow(2)).sqrt()
        
        # Modulus obtained as modulus of LR components
        data_lr_u = self.spatial_downsample_interpolate(data_hr_u)
        data_lr_v = self.spatial_downsample_interpolate(data_hr_v)
        data_lr_gt = (data_lr_u.pow(2) + data_lr_v.pow(2)).sqrt()
        
        # Alternative : persistence models
        if True:
            data_lr_obs   = self.get_persistence(data_lr_gt, 'lr', longer_series = True)
            data_hr_obs   = self.get_persistence(data_hr_gt, 'hr', longer_series = True)
        else:
            data_lr_obs   = data_lr_gt
            data_hr_obs   = data_hr_gt
        #end
        
        # NOTE: is in-situ time series are actually measured, these positions
        # in the persistence model must be filled with in-situ time series
        # These loops are not nice but necessary for the experiment on buoys shuffling
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
        wind_hist   = wind_hist[:, timewindow_start : timewindow_end, :,:]
        
        if True:
            # This modification makes persistence and naive initializations to match
            # That is, we assume to "close" each time series with u(23h, day0) = u(0h, day1)
            # Activate it to implemen this (then initializations with persistence and naive match)
            data_lr_obs[:,-1,:,:] = data_lr_gt[:,-1,:,:]
        #end
        
        # Temporal interpolation
        # data_lr_obs = self.interpolate_channelwise(data_lr_obs, timesteps)

        return data_lr_gt, data_lr_obs, data_hr_gt, data_an_obs, data_hr_obs, wind_hist
    #end
    
    def compute_loss(self, data, batch_idx, iteration, phase = 'train', init_state = None):
        
        wind_lr_gt, wind_lr, wind_hr_gt, wind_an, wind_hr, wind_hist_gt = self.prepare_batch(data)
        
        # Mask data
        mask, mask_lr, mask_hr_dx1,_ = self.get_osse_mask(wind_hr.shape)
        
        # Concatenate low-resolution and anomaly (wind fields) and apply mask
        batch_input = torch.cat([wind_lr, wind_an, wind_an], dim = 1)
        batch_input = batch_input * mask
        
        # Inversion
        with torch.set_grad_enabled(True):
            batch_input = torch.autograd.Variable(batch_input, requires_grad = True)
            
            if self.hparams.inversion == 'fp':
                
                output = self.model(batch_input)
                reco_lr = self.interpolate_channelwise(wind_lr.mul(mask_lr))
                reco_an = output[:,48:,:,:]
                reco_hr = reco_lr + reco_an
                wind_hist_out = self.h_Phi(reco_hr * self.normparams['std'])
                
            elif self.hparams.inversion == 'gs':
                
                output,_,_,_ = self.model(batch_input, batch_input, mask)
                reco_lr = output[:,:24,:,:]
                reco_an = output[:,48:,:,:]
                reco_hr = (reco_lr + reco_an) * self.normparams['std']
                wind_hist_out = self.h_Phi(reco_hr)
            #end
        #end
        
        # Transform output in exp, if necessary
        wind_hist_out = wind_hist_out.exp()
        
        # Save reconstructions
        # Denormalization does not take effect if normalize has been set to False
        # when initializing the datamodule. std is set to 1 in that case
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            self.save_samples({
                'data'  : wind_hist_gt.detach().cpu(),
                'reco'  : wind_hist_out.detach().cpu(),
                'wdata' : wind_hr_gt.detach().cpu() * self.normparams['std'],
                'wreco' : reco_hr.detach().cpu() * self.normparams['std']
            })
        #end
        
        if self.current_epoch in [0, 24, 49]:
            batch_hist = {'data' : wind_hist_gt.detach().cpu(), 'reco' : wind_hist_out.detach().cpu()}
            torch.save( batch_hist, open(f'./postprocessing/histograms-epoch-{self.current_epoch}.pkl', 'wb') )
        #end
        
        # Compute loss
        loss = self.l2_loss((wind_hist_out - wind_hist_gt), mask = None)
        
        if False:
            loss += self.l2_loss((wind_hr_gt - reco_hr), mask = None)
            grad_data = torch.gradient(wind_hr_gt, dim = (3,2))
            grad_reco = torch.gradient(reco_hr, dim = (3,2))
            loss_grad_x = self.loss_fn((grad_data[1] - grad_reco[1]), mask = None)
            loss_grad_y = self.loss_fn((grad_data[0] - grad_reco[0]), mask = None)
            loss += (loss_grad_x + loss_grad_y) * 1.
        #end
        
        # Monitor Hellinger Distance
        hdistance = self.hd_loss(wind_hist_gt.detach().cpu(), wind_hist_out.detach().cpu())
        
        return dict({'loss' : loss, 'hdistance' : hdistance}), wind_hist_out, hdistance
    #end
#end
