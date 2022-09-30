
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import solver as NN_4DVar

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

class RBlock(nn.Module):
    def __init__(self):
        super(RBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Conv2d(72, 50, (3,3), padding = 1, stride = 1, bias = False),
            nn.BatchNorm2d(100),
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


class Phi_r(nn.Module):
    def __init__(self, shape_data, config_params):
        super(Phi_r, self).__init__()
        
        self.rnet = nn.Sequential(
            RBlock()
        )
    #end
    
    def forward(self, data):
        
        return self.rnet(data)
    #end
#end


class Phi(nn.Module):
    ''' Dynamical prior '''
    
    def __init__(self, shape_data, config_params):
        super(Phi, self).__init__()
        	
        ts_length = shape_data[1] * 3
        img_H, img_W = shape_data[-2:]
        
        self.net = nn.Sequential(
            nn.Conv2d(ts_length, 50, (3,3),
                      padding = 'same', padding_mode = 'reflect', bias = False),
            nn.BatchNorm2d(20),
            nn.LeakyReLU(0.1),
            nn.Conv2d(50, ts_length, (3,3),
                      padding = 'same', bias = False
            )
        )
    #end
    
    def forward(self, data):
        
        reco = self.net(data)
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
        
        obs_term = (x - y_obs).mul(mask)
        return obs_term
    #end
#end


class _NormLoss(nn.Module):
    
    def __init__(self, dim_item = 2):
        super(_NormLoss, self).__init__()
        
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


class NormLoss(nn.Module):
    
    def __init__(self):
        super(NormLoss, self).__init__()
        
    #end
    
    def forward(self, item, mask):
        
        if mask is None:
            mask = torch.ones_like(item)
        #end
        
        # square
        argument = item.pow(2)
        argument = argument.mul(mask)
        
        if mask.sum() == 0.:
            n_items = 1.
        else:
            n_items = mask.sum().div(24.)
        #end
        
        # sum on: 
        #   1. features plane; 
        #   2. timesteps and batches
        # Then mean over effective items
        argument = argument.sum(dim = (2,3))
        argument = argument.sum(dim = (1,0))
        loss = argument.div(n_items)
        
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
        self.hparams.lr_kernel_size         = 31
        self.hparams.lr_padding             = 0
        self.hparams.lr_stride              = 1
        self.hparams.fixed_point            = config_params.FIXED_POINT
        self.hparams.hr_mask_mode           = config_params.HR_MASK_MODE
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
    
    def avgpool2d_keepsize(self, data):
        
        img_size = data.shape[-2:]
        pooled = F.avg_pool2d(data, 
                              kernel_size = self.hparams.lr_kernel_size,
                              padding = self.hparams.lr_padding, 
                              stride = self.hparams.lr_stride)
        pooled  = F.interpolate(pooled, size = tuple(img_size), mode = 'bicubic')
        
        if not data.shape == pooled.shape:
            raise ValueError('Original and Pooled_keepsize data shapes mismatch')
        #end
        
        return pooled
    #end
    
    def get_hr_local_observation(self, data):
        
        center_x, center_y = data.shape[-2] // 2, data.shape[-1] // 2
        return data[:,:, center_x, center_y].unsqueeze(-1).unsqueeze(-1)
    #end
    
    def get_mask(self, data_shape, mode):
        
        if mode == 'center':
            
            center_h, center_w = data_shape[-2] // 2, data_shape[-1] // 2
            mask = torch.zeros(data_shape)
            mask[:,:, center_h, center_w] = 1.
            
        elif mode == 'patch':
            
            delta_x = 10
            center_h, center_w = data_shape[-2] // 2, data_shape[-1] // 2
            mask = torch.zeros(data_shape)
            mask[:,:, center_h - delta_x : center_h + delta_x, 
                 center_w - delta_x : center_w + delta_x] = 1.
            
        elif mode == 'zeroes':
            
            mask = torch.zeros(data_shape)
            
        elif mode == 'points':
            
            mask = torch.zeros(data_shape)
            points_x = [4, 4, 30, 30]
            points_y = [4, 30, 4, 30]
            mask[:,:,points_x, points_y] = 1.
        else:
            raise ValueError('Mask mode not impletemented.')
        #end
        
        return mask
    #end
    
    def forward(self, batch, phase = 'train'):
        
        state_init = None
        for n in range(self.hparams.n_fourdvar_iter):
            
            loss, outs = self.compute_loss(batch, iteration = n, phase = phase, init_state = state_init)
            state_init = outs.detach()
        #end
        
        return loss, outs
    #end
    
    def compute_loss(self, data, iteration, phase = 'train', init_state = None):
        
        # Prepare input data
        data_hr = data.clone()
        data_lr = self.avgpool2d_keepsize(data_hr)
        input_data = torch.cat((data_lr, data_hr - data_lr, data_hr - data_lr), dim = 1)
                
        # Prepare input state initialized
        if init_state is None:
            input_state = torch.cat((data_lr, data_hr - data_lr, data_hr - data_lr), dim = 1)
        else:
            input_state = init_state
        #end
        
        # Mask data
        mask_lr = torch.ones_like(data_lr)
        mask_hr = self.get_mask(data_hr.shape, mode = self.hparams.hr_mask_mode)
        mask = torch.cat((mask_lr, mask_hr, torch.zeros_like(data_hr)), dim = 1)
        
        input_state = input_state * mask
        input_data  = input_data * mask
        
        with torch.set_grad_enabled(True):
            input_state = torch.autograd.Variable(input_state, requires_grad = True)
            
            if self.hparams.fixed_point:
                outputs = self.Phi(input_data)
                reco_lr = data_lr.clone()
                reco_hr = outputs[:,48:,:,:]
                reco = data_lr + reco_hr
            else:
                outputs, _,_,_ = self.model(input_state, input_data, mask)
                reco_lr = outputs[:,:24,:,:]
                reco_hr = outputs[:,48:,:,:]
                reco = reco_lr + reco_hr
            #end
        #end
        
        # Save reconstructions
        if phase == 'test' and iteration == self.hparams.n_fourdvar_iter-1:
            self.save_samples({'data' : data.detach().cpu(), 
                               'reco' : reco.detach().cpu()})
        #end
        
        # mask_loss = self.get_mask(data_lr.shape, mode = 'patch')
        
        # Return loss, computed as reconstruction loss
        # ''' NOTE : loss function is NormLoss but in these data 
        #     there are no missing values!!! '''
        # reco_lr = outputs[:,:24,:,:]
        # reco_hr = outputs[:,48:,:,:]
        # reco = ( reco_lr + reco_hr )
        
        loss_lr = self.loss_fn( (reco_lr - data_lr), mask = None )
        loss_hr = self.loss_fn( (reco - data_hr), mask = None )
        loss = self.hparams.weight_lres * loss_lr + self.hparams.weight_hres * loss_hr
                
        grad_data = torch.gradient(data_hr, dim = (3,2))
        grad_reco = torch.gradient(reco, dim = (3,2))
        
        loss_grad_x = self.loss_fn( (grad_data[0] - grad_reco[0]), mask = None )
        loss_grad_y = self.loss_fn( (grad_data[1] - grad_reco[1]), mask = None )
        loss_grad = loss_grad_x + loss_grad_y
        
        loss += loss_grad * 1e-3
        
        regularization = self.loss_fn( (outputs - self.Phi(outputs)), mask = None )
        loss += regularization * 1e-2
        #end
        
        return dict({'loss' : loss}), outputs
    #end
    
    def training_step(self, batch, batch_idx):
        
        metrics, out = self.forward(batch, phase = 'train')
        loss = metrics['loss']
        self.log('loss', loss, on_step = True, on_epoch = True, prog_bar = True)
        
        return loss
    #end
    
    def training_epoch_end(self, outputs):
        
        loss = torch.stack([out['loss'] for out in outputs]).mean()
        self.save_epoch_loss(loss, self.current_epoch, 'train')
    #end
    
    def validation_step(self, batch, batch_idx):
        
        metrics, out = self.forward(batch, phase = 'train')
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
            metrics, outs = self.forward(batch, phase = 'test')
            
            test_loss = metrics['loss']
            self.log('test_loss', test_loss.item())
        #end
        
        self.save_test_loss(test_loss, batch.shape[0])
        return metrics, outs
    #end
    
    def get_eval_metrics(self):
        
        data_reco = self.get_saved_samples()
        
        data = torch.cat([item['data'] for item in data_reco], dim = 0)
        reco = torch.cat([item['reco'] for item in data_reco], dim = 0)
        # reco_lr = reco[:,:24,:,:]
        # reco_hr = reco[:,48:,:,:]
        # reco = reco_lr + reco_hr
        cp_data = crop_central_patch(data, length = 10)
        cp_reco = crop_central_patch(reco, length = 10)
        
        hist_data = get_batched_histograms(data, bins = 30)
        hist_reco = get_batched_histograms(reco, bins = 30)
        cp_hist_data = get_batched_histograms(cp_data, bins = 30)
        cp_hist_reco = get_batched_histograms(cp_reco, bins = 30)
        
        mse_complete = NormLoss()((data - reco), mask = None)
        mse_central  = NormLoss()((cp_data - cp_reco), mask = None)
        bdist_complete = bhattacharyya_distance(hist_data, hist_reco)
        bdist_central = bhattacharyya_distance(cp_hist_data, cp_hist_reco)
        
        perf_metrics_dict = {
            'mse_central'  : mse_central.item(),
            'mse_complete' : mse_complete.item(),
            'bd_central'   : bdist_central.item(),
            'bd_complete'  : bdist_complete.item()
        }
        return perf_metrics_dict
    #end
    
#end


# Define evaluation metrics : Bhattacharyya distance, MSE

def bhattacharyya_distance(h_target, h_output, reduction_dim = 1, mode = 'trineq'):
    
    if reduction_dim is None and h_target.shape.__len__() > 1:
        reduction_dim = 1
    elif reduction_dim is None and h_target.shape.__len__() <= 1:
        reduction_dim = 0
    #end
    
    eps = 1e-5
    b_coefficient = torch.sum((h_target * h_output).sqrt(), dim = reduction_dim)
    if torch.any(b_coefficient > 1) or torch.any(b_coefficient < 0):
        raise ValueError('BC can not be > 1 or < 0')
    #end
    
    b_coefficient[b_coefficient < eps] = eps
    
    if mode == 'log':
        b_distance = -1. * torch.log(b_coefficient)
    elif mode == 'trineq':
        b_distance = torch.sqrt(1 - b_coefficient)
    else:
        raise NotImplementedError('Metric selected not valid. Consider setting "log" or "trineq"')
    #end
    
    return b_distance.mean()
#end

def mse_expl_variance(target, output, divide_std = True):
    
    mserror = (target - output).pow(2).mean(dim = (2,3)).sum(dim = 1).mean()
    if divide_std:
        mserror = mserror / target.std()
    #end
    
    return mserror
#end

# Crop central patch
def crop_central_patch(img_ts, length = 10):
    
    center_h, center_w = img_ts[0,0].shape[-2] // 2, img_ts[0,0].shape[-1] // 2
    cp_img = img_ts[:,:, center_h - length : center_h + length, center_w - length : center_w + length]
    return cp_img
#end

# B-distances
def get_batched_histograms(img_ts, bins = 30):
    
    img_H, img_W = img_ts[0,0].shape
    img = img_ts.reshape(-1, img_H, img_W)
    hist = torch.cat([torch.histc(img[i], bins = bins).unsqueeze(0) for i in range(img.shape[0])], dim = 0)
    for i in range(hist.shape[0]):
        hist[i] = hist[i] / hist[i].sum()
    #end
    return hist
#end
