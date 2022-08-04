
import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
import solver as NN_4DVar


class Phi_r(nn.Module):
    ''' Dynamical prior '''
    
    def __init__(self, shape_data, config_params):
        super(Phi_r, self).__init__()
        
        ts_length = shape_data[1]
        
        # Conv2D-AE
        self.encoder = nn.Sequential(
            nn.Conv2d(ts_length, 48, (10,10), padding = 1),
            nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
            nn.Conv2d(48, 72, (10,10), padding = 1),
            nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
            nn.Conv2d(72, 128, (6,6),  padding = 1)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 72, (6,6),  padding = 1),
            nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
            nn.ConvTranspose2d(72, 48, (10,10), padding = 1),
            nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
            nn.ConvTranspose2d(48, ts_length, (10,10), padding = 1),
            nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
        )
    #end
    
    def forward(self, data):
        
        latent = self.encoder(data)
        reco = self.decoder(latent)
        return reco
    #end
#end


class ObsModel_Mask(nn.Module):
    ''' Observation model '''
    
    def __init__(self, shape_data, dim_obs):
        super(ObsModel_Mask, self).__init__()
        
        self.shape_data = shape_data
        self.dim_obs    = dim_obs
        self.mask_obs   = torch.zeros(shape_data)
        self.downsample = nn.AvgPool2d(6,6)
        self.dim_obs_channel = np.array([shape_data[0], dim_obs])
    #end
    
    def get_mask(self, img):
        
        height, width = img.shape[-2:]
        height_center, width_center = height // 2, width // 2
        mask = torch.zeros_like(img)
        mask[:,:, height_center, width_center] = 1.
        return mask
    #end
    
    def forward(self, y_obs, x, mask):
        
        mask = self.get_mask(y_obs)
        y_lr = self.downsample(y_obs)
        obs_term_mask = (y_obs - x).mul(mask)
        obs_term_lr   = (y_lr - self.downsample(x))
        return obs_term_mask, obs_term_lr
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
        batch_size, height, width, ts_length = shape_data
        mgrad_shapedata = [height, width, ts_length]
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
            shape_data,                              # Shape data
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
        
        losses, bsizes = torch.Tensor(self.__test_loss), torch.Tensor(self.__test_batches_size)
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
            self.__vel_losses[epoch] = loss.item()
        #end
    #end
    
    def get_learning_curves(self):
        
        return self.__train_losses, self.__val_losses
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
    
    def compute_loss(self, data, phase = 'train'):
        
        # Prepare input data
        input_data = data.clone()
        mask = None
        
        # Prepare input state initialized
        input_state = torch.zeros_like(input_data)
        
        # with torch.set_grad_enabled(True): 
        # inputs_init = torch.autograd.Variable(inputs_init, requires_grad = True)
        # outputs, _,_,_ = self.model(inputs_init, data_input, mask_input)
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            outputs, _,_,_ = self.model(input_state, input_data, mask)
        #end
        
        # Save reconstructionss
        if phase == 'test':
            self.save_samples({'data' : data.detach().cpu(), 
                               'reco' : outputs.detach().cpu()})
        #end
        
        # Return loss, computed as reconstruction loss
        loss = self.loss_fn( (outputs - data), mask = None )
        
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
