
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import solver as NN_4DVar


class Phi_r(nn.Module):
    ''' Dynamical prior '''
    
    def __init__(self, shape_data, config_params):
        super(Phi_r, self).__init__()
        	
        ts_length = shape_data[1] * 2
        
        self.net = nn.Sequential(
            nn.Conv2d(ts_length, ts_length, kernel_size = (6,6), padding = 0),
            nn.ReLU(),
            nn.ConvTranspose2d(ts_length, ts_length, kernel_size = (6,6), padding = 0)
        )
        
        # Conv2D-AE
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(ts_length, 48, (10,10), padding = 1),
        #     nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
        #     nn.Conv2d(48, 72, (10,10), padding = 1),
        #     nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
        #     nn.Conv2d(72, 128, (6,6),  padding = 1)
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(128, 72, (6,6),  padding = 1),
        #     nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
        #     nn.ConvTranspose2d(72, 48, (10,10), padding = 1),
        #     nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
        #     nn.ConvTranspose2d(48, ts_length, (10,10), padding = 1),
        #     nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
        # )
    #end
    
    def forward(self, data):
        
        # latent = self.encoder(data)
        # reco = self.decoder(latent)
        reco = self.net(data)
        return reco
    #end
#end


class ObsModel_Mask(nn.Module):
    ''' Observation model '''
    
    def __init__(self, shape_data, dim_obs):
        super(ObsModel_Mask, self).__init__()
        
        self.shape_data = shape_data
        self.dim_obs = dim_obs
        self.dim_obs_channel = np.array([shape_data[0], dim_obs])
    #end
    
    def forward(self, x, y_obs, mask):
        
        y_lr, y_pt_hr = y_obs
        center_h, center_w = y_lr.shape[-2] // 2, y_lr.shape[-1] // 2
        
        obs_term_lr = (x - torch.cat([y_lr, y_lr], dim = 1))
        obs_term_hr = (x[:,:, center_h, center_w] - y_pt_hr)
        return obs_term_lr, obs_term_hr
    #end
#end

class NormLoss(nn.Module):
    
    def __init__(self, dim_item = 2):
        super(NormLoss, self).__init__()
        
        self.dim_item = dim_item
    #end
    
    def forward(self, item, mask):
        
        # square
        argument = item.pow(2)
        if mask is not None:
            argument = argument.mul(mask)
        #end
        
        # feature-wise norm
        if self.dim_item == 2:
            argument = argument.mean(dim = (-2, -1))
        elif self.dim_item == 1:
            argument = argument.mean(dim = -1)
        #end
        
        # sum over time steps and batch-wise mean
        argument = argument.sum(dim = -1)
        loss = argument.mean()
        
        return loss
    #end
#end


class LitModel(pl.LightningModule):
    
    def __init__(self, Phi, shape_data, config_params):
        super(LitModel, self).__init__()
        
        # Dynamical prior
        self.Phi = Phi
        
        # Loss function — parameters optimization
        self.loss_fn = NormLoss()
        
        # Hyper-parameters, learning and workflow
        self.hparams.kernel_size            = (6,6) # tuple(config_params.KERNEL_SIZE)
        self.hparams.padding                = 0     # config_params.PADDING
        self.hparams.stride                 = (6,6) # tuple(config_params.STRIDE)
        self.hparams.weight_hr              = config_params.WEIGHT_HR
        self.hparams.weight_lr              = config_params.WEIGHT_LR
        self.hparams.mgrad_lr               = config_params.SOLVER_LR
        self.hparams.mgrad_wd               = config_params.SOLVER_WD
        self.hparams.prior_lr               = config_params.PHI_LR
        self.hparams.prior_wd               = config_params.PHI_WD
        self.hparams.dim_grad_solver        = config_params.DIM_LSTM
        self.hparams.dropout                = config_params.SOL_DROPOUT
        self.hparams.n_solver_iter          = config_params.NSOL_ITER
        self.hparams.n_fourdvar_iter        = config_params.N_4DV_ITER
        self.hparams.automatic_optimization = True
        
        # Monitoring metrics
        self.__train_losses = np.zeros(config_params.EPOCHS)
        self.__val_losses   = np.zeros(config_params.EPOCHS)
        
        # Save reconstructions to a list—protected fields
        self.__samples_to_save = list()
        self.__test_loss = list()
        self.__test_batches_size = list()
        
        # Initialize gradient solver (LSTM)
        batch_size, ts_length, height, width = shape_data
        mgrad_shapedata = [ts_length * 2, height, width]
        model_shapedata = [batch_size, ts_length, height, width]
        alpha_obs = config_params.ALPHA_OBS
        alpha_reg = config_params.ALPHA_REG
        
        self.model = NN_4DVar.Solver_Grad_4DVarNN(
            self.Phi,                                # Prior
            ObsModel_Mask(shape_data, dim_obs = 2),  # Observation model
            NN_4DVar.model_GradUpdateLSTM(           # Gradient solver
                mgrad_shapedata,                       # m_Grad : Shape data
                False,                                 # m_Grad : Periodic BCs
                self.hparams.dim_grad_solver,          # m_Grad : Dim LSTM
                self.hparams.dropout,                  # m_Grad : Dropout
            ),
            NormLoss(dim_item = 2),                  # Norm Observation
            NormLoss(dim_item = 2),                  # Norm Prior
            model_shapedata,                         # Shape data
            self.hparams.n_fourdvar_iter,            # Solver iterations
            alphaObs = alpha_obs,                    # alpha observations
            alphaReg = alpha_reg                     # alpha regularization
        )
    #end
    
    def save_test_loss(self, test_loss, batch_size):
                
        self.__test_loss.append(test_loss)
        self.__test_batches_size.append(batch_size)
    #end
    
    def get_test_loss(self):
        
        losses = torch.Tensor(self.__test_loss)
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
    
    def forward(self, data):
        
        print('Forward')
        loss, out = self.compute_loss(data)
        return loss, out
    #end
    
    def configure_optimizers(self):
        
        optimizers = torch.optim.Adam(
            [
                {'params'       : self.model.model_Grad.parameters(),
                 'lr'           : self.hparams.mgrad_lr,
                 'weight_decay' : self.hparams.mgrad_wd},
                {'params'       : self.model.Phi.parameters(),
                 'lr'           : self.hparams.prior_lr,
                 'weight_decay' : self.hparams.prior_wd}
            ]
        )
        return optimizers
    #end
    
    def avgpool2d_keepsize(self, data, kernel_size, padding, stride):
        
        img_size = data.shape[-2:]
        pooled = F.avg_pool2d(data, kernel_size = kernel_size,
                              padding = padding, stride = stride)
        pooled  = F.interpolate(pooled, size = tuple(img_size), mode = 'nearest')
        
        if not data.shape == pooled.shape:
            raise ValueError('Original and Pooled_keepsize data shapes mismatch')
        #end
        
        return pooled
    #end
    
    def get_hr_local_observation(self, data):
        
        center_x, center_y = data.shape[-2] // 2, data.shape[-1] // 2
        return data[:,:, center_x, center_y].unsqueeze(-1).unsqueeze(-1)
    #end
    
    def compute_loss(self, data, phase = 'train'):
        
        # Prepare input data
        data_hr = data.clone()
        data_lr = self.avgpool2d_keepsize(data_hr, 
                                          kernel_size = self.hparams.kernel_size, 
                                          padding = self.hparams.padding, 
                                          stride = self.hparams.stride)
        local_hr = self.get_hr_local_observation(data_hr)
        mask = None
        
        # Prepare input state initialized
        # input_state = torch.zeros_like(data_lr)
        input_state = torch.cat(
            [data_lr,                     # Low-resolution component
             torch.zeros_like(data_lr)],  # Anomaly component
            dim = 1)
        
        # with torch.set_grad_enabled(True): 
        # inputs_init = torch.autograd.Variable(inputs_init, requires_grad = True)
        # outputs, _,_,_ = self.model(inputs_init, data_input, mask_input)
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            outputs, _,_,_ = self.model(input_state, [data_lr, local_hr], mask)
        #end
        
        # Save reconstructions
        if phase == 'test':
            print(outputs.shape)
            self.save_samples({'data' : data.detach().cpu(), 
                               'reco' : outputs.detach().cpu()})
        #end
        
        # Return loss, computed as reconstruction loss
        # loss_lr = self.loss_fn( (outputs[:,:24,:,:] - data_lr), mask = None )
        # loss_hr = self.loss_fn( (outputs[:,24:,:,:] - data_hr), mask = None )
        
        reco_lr_plus_hr = outputs[:,:24,:,:] + outputs[:,24:,:,:]
        loss = self.loss_fn((reco_lr_plus_hr - data_hr), mask = None)
        
        # loss = self.hparams.weight_lr * loss_lr + self.hparams.weight_hr * loss_hr               
        
        return dict({'loss' : loss}), outputs
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, out = self.compute_loss(batch, phase = 'train')
        loss = metrics['loss']
        self.log('loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        
        return loss
    #end
    
    def training_epoch_end(self, outputs):
        
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.save_epoch_loss(loss, self.current_epoch, 'train')
    #end
    
    def validation_step(self, batch, batch_idx):
        
        metrics, out = self.compute_loss(batch, phase = 'train')
        val_loss = metrics['loss']
        self.log('val_loss', val_loss)
        
        return val_loss
    #end
    
    def validation_epoch_end(self, outputs):
        
        loss = torch.stack([out for out in outputs]).mean()
        self.save_epoch_loss(loss, self.current_epoch, 'val')
    #end
    
    def test_step(self, batch, batch_idx):
        
        with torch.no_grad():
            metrics, outs = self.compute_loss(batch, phase = 'test')
            
            test_loss = metrics['loss']
            self.log('test_loss', test_loss.item())
        #end
        
        self.save_test_loss(test_loss, batch.shape[0])
        return metrics, outs
    #end
    
#end
