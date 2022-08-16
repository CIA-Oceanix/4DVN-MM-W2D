
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import solver as NN_4DVar


# class Phi(nn.Module):
#     ''' Dynamical prior '''
    
#     def __init__(self, shape_data, config_params):
#         super(Phi, self).__init__()
        	
#         ts_length = shape_data[1] * 2
        
#         if config_params.PRIOR == 'CL':
            
#             # 1 couche conv
#             self.prior = 'cl'
#             self.net = nn.Sequential(
#                 nn.Conv2d(ts_length, ts_length, kernel_size = (6,6), padding = 'same'),
#                 nn.ReLU(),
#                 nn.ConvTranspose2d(ts_length, ts_length, kernel_size = (6,6), padding = 'same')
#             )
        
#         elif config_params.PRIOR == 'AE':
            
#             # Conv2D-AE
#             self.prior = 'ae'
#             self.encoder = nn.Sequential(
#                 nn.Conv2d(ts_length, 72, (3,3), padding = 0),
#                 nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU()
#             )
#             self.decoder = nn.Sequential(
#                 nn.ConvTranspose2d(72, ts_length, (3,3), padding = 0)#,
#                 # nn.Dropout(config_params.PHI_DROPOUT), nn.ReLU(),
#             )
        
#         else:
            
#             raise NotImplementedError('No valid prior chosen')
#         #end
#     #end
    
#     def forward(self, data):
        
#         if self.prior == 'cl':
#             reco = self.net(data)
#             return reco
        
#         elif self.prior == 'ae':
#             latent = self.encoder(data)
#             reco = self.decoder(latent)
#             return reco
            
#         else:
#             raise NotImplementedError('No valid prior chosen')
#         #end
#     #end
# #end


# class Encoder(nn.Module):
#     def __init__(self, shape_data, config_params):
#         super(Encoder, self).__init__()
        
#         # encoder
#         self.enc_avgpool1 = nn.AvgPool2d(3)
#         self.enc_conv1 = nn.Conv2d(48, 2 * 48, (3,3), padding = 0)
#         self.enc_conv2 = nn.Conv2d(2 * 48, 3 * 48, (3,3), padding = 0)
        
#         # decoder
#         # lr net
#         self.dec_convt1_lr = nn.ConvTranspose2d(3 * 48, 2 * 48, (3,3), padding = 3, stride = 3)
#         self.dec_convt2_lr = nn.ConvTranspose2d(2 * 48, 48, (3,3), padding = 3, stride = 3)
#         self.dec_convt3_lr = nn.ConvTranspose2d(48, 24, (3,3), padding = 3)
        
#         # hr net
#         self.dec_convt1_hr = nn.ConvTranspose2d(3 * 48, 2 * 48, (3,3), padding = 3, stride = 3)
#         self.dec_convt2_hr = nn.ConvTranspose2d(2 * 48, 48, (3,3), padding = 3, stride = 3)
#         self.dec_convt3_hr = nn.ConvTranspose2d(48, 24, (3,3), padding = 3)
#     #end
    
#     def forward(self, data):
        
#         # print('Data : {}'.format(data.shape))
#         #encoder
#         data = self.enc_avgpool1(data)
#         latent = self.enc_conv1(data)
#         latent = self.enc_conv2(latent)
#         # print('latent : {}'.format(latent.shape))
        
#         # decoder lr
#         reco = self.dec_convt1_lr(F.relu(latent))
#         reco = self.dec_convt2_lr(F.relu(reco))
#         reco_lr = self.dec_convt3_lr(F.relu(reco))
#         # print('LR output : {}'.format(reco_lr.shape))
        
#         # decoder hr
#         reco = self.dec_convt1_hr(F.relu(latent))
#         reco = self.dec_convt2_hr(F.relu(reco))
#         reco_hr = self.dec_convt3_hr(F.relu(reco))
#         # print('HR output : {}'.format(reco_hr.shape))
        
#         reco = torch.cat((reco_lr, reco_hr), dim = 1)
#         # print(reco.shape)
#         return reco
#     #end
# #end


class BiLinUnit(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dim, dw, dw2, dropout=0.):
        super(BiLinUnit, self).__init__()
        self.conv1 = torch.nn.Conv2d(dim_in, 2 * dim, (2 * dw + 1, 2 * dw + 1), padding=dw, bias=False)
        self.conv2 = torch.nn.Conv2d(2 * dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim, dim_out, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.bilin0 = torch.nn.Conv2d(dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.bilin1 = torch.nn.Conv2d(dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.bilin2 = torch.nn.Conv2d(dim, dim, (2 * dw2 + 1, 2 * dw2 + 1), padding=dw2, bias=False)
        self.dropout = torch.nn.Dropout(dropout)
    #end
    
    def forward(self, xin):
        x = self.conv1(xin)
        x = self.dropout(x)
        x = self.conv2(F.relu(x))
        x = self.dropout(x)
        x = torch.cat((self.bilin0(x), self.bilin1(x) * self.bilin2(x)), dim=1)
        x = self.dropout(x)
        x = self.conv3(x)
        return x
    #end
#end

class Encoder(torch.nn.Module):
    def __init__(self, shape_data, config_params,
                 dim_inp     = 48,
                 dim_out     = 24,
                 dim_ae      = 24,
                 dw          = 6,
                 dw2         = 6,
                 ss          = 5,
                 nb_blocks   = 3,
                 rateDropout = 0.):
        super(Encoder, self).__init__()
        
        self.nb_blocks = nb_blocks
        self.dim_ae = dim_ae
        self.pool1 = torch.nn.AvgPool2d(ss)
        print(dim_inp, dim_out, dim_ae, dw, dw2, ss, nb_blocks, rateDropout)
        self.conv_tr = torch.nn.ConvTranspose2d(dim_out, dim_out, (ss, ss),
                                                stride = (ss, ss),
                                                bias = False)
        
        self.nn_lr = self.__make_BilinNN(dim_inp, dim_out, self.dim_ae, dw, dw2, self.nb_blocks, rateDropout)
        self.nn_hr = self.__make_BilinNN(dim_inp, dim_out, self.dim_ae, dw, dw2, self.nb_blocks, rateDropout)
        self.dropout = torch.nn.Dropout(rateDropout)
    #end
    
    def __make_BilinNN(self, dim_inp, dim_out, dim_ae, dw, dw2, 
                       nb_blocks = 2,
                       dropout = 0.):
        layers = []
        layers.append(BiLinUnit(dim_inp, dim_out, dim_ae, dw, dw2, dropout))
        for kk in range(0, nb_blocks - 1):
            layers.append(BiLinUnit(dim_ae, dim_out, dim_ae, dw, dw2, dropout))
        return torch.nn.Sequential(*layers)
    #end
    
    def forward(self, xinp):
        ## LR component
        x_lr = self.nn_lr(self.pool1(xinp))
        x_lr = self.dropout(x_lr)
        
        # HR component
        x_hr = self.nn_hr(xinp)
        
        print(x_lr.shape, x_hr.shape)
        # return x_lr + x_hr
        return torch.cat((x_lr, x_hr), dim = 1)
    #end
#end


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
    #end
    
    def forward(self, data):
        
        return torch.mul(1., data)
    #end
#end


class Phi_r(nn.Module):
    def __init__(self, shape_data, config_params):
        super(Phi_r, self).__init__()
        
        self.encoder = Encoder(shape_data, config_params)
        self.decoder = Decoder()
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
        
        # NOTE : chennels == time series length
        self.shape_data = shape_data
        self.dim_obs = dim_obs
        self.dim_obs_channel = np.array([shape_data[1], dim_obs])
    #end
    
    def forward(self, x, y_obs, mask):
        
        return (x - y_obs).mul(mask)
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
        self.hparams.weight_hres            = config_params.WEIGHT_HRES
        self.hparams.weight_lres            = config_params.WEIGHT_LRES
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
            ObsModel_Mask(shape_data, dim_obs = 1),  # Observation model
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
        pooled  = F.interpolate(pooled, size = tuple(img_size), mode = 'bilinear')
        
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
        input_data = torch.cat([data_lr, data_hr], dim = 1)
        
        # Prepare input state initialized
        # input_state = torch.zeros_like(data_lr)
        input_state = torch.cat(
            [data_lr,                     # Low-resolution component
             torch.zeros_like(data_lr)],  # Anomaly component
            dim = 1)
        
        # Mask data
        center_h, center_w = data.shape[-2] // 2, data.shape[-1] // 2
        mask = torch.zeros_like(input_state)
        mask[:,:24,:,:] = 1.
        mask[:,24:, center_h, center_w] = 1.
        
        input_state = input_state * mask
        
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            outputs, _,_,_ = self.model(input_state, input_data, mask)
        #end
        
        # Save reconstructions
        if phase == 'test':
            print(outputs.shape)
            print(outputs[:,:24,:,:].mean())
            print(outputs[:,24:,:,:].mean())
            self.save_samples({'data' : data.detach().cpu(), 
                               'reco' : outputs.detach().cpu()})
        #end
        
        # Return loss, computed as reconstruction loss
        reco_lr = outputs[:,:24,:,:]
        reco_tot = outputs[:,:24,:,:] + outputs[:,24:,:,:]
        loss_lr = self.loss_fn( (reco_lr - data_lr), mask = None )
        loss_hr = self.loss_fn( (reco_tot - data_hr), mask = None )
        
        # autres terms : || x - Phi(x) || 
        
        loss = self.hparams.weight_lres * loss_lr + self.hparams.weight_hres * loss_hr               
        
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
