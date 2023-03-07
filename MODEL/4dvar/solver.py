#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:38:05 2020
@author: rfablet
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cpu')
#end


class CorrelateNoise(torch.nn.Module):
    def __init__(self, shape_data, dim_cn):
        super(CorrelateNoise, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_cn, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_cn, 2 * dim_cn, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_cn, shape_data, (3, 3), padding=1, bias=False)
    #end
    
    def forward(self, w):
        w = self.conv1(F.relu(w)).to(DEVICE)
        w = self.conv2(F.relu(w)).to(DEVICE)
        w = self.conv3(w).to(DEVICE)
        return w
    #end
#end

class RegularizeVariance(torch.nn.Module):
    def __init__(self, shape_data, dim_rv):
        super(RegularizeVariance, self).__init__()
        self.conv1 = torch.nn.Conv2d(shape_data, dim_rv, (3, 3), padding=1, bias=False)
        self.conv2 = torch.nn.Conv2d(dim_rv, 2 * dim_rv, (3, 3), padding=1, bias=False)
        self.conv3 = torch.nn.Conv2d(2 * dim_rv, shape_data, (3, 3), padding=1, bias=False)
    #end    
    
    def forward(self, v):
        v = self.conv1(F.relu(v)).to(DEVICE)
        v = self.conv2(F.relu(v)).to(DEVICE)
        v = self.conv3(v).to(DEVICE)
        return v
    #end
#end

def compute_WeightedLoss(x2,w):
    x2_msk = x2[:, w==1, ...]
    x2_num = ~x2_msk.isnan() & ~x2_msk.isinf()
    loss2 = F.mse_loss(x2_msk[x2_num], torch.zeros_like(x2_msk[x2_num]))
    loss2 = loss2 *  w.sum()
    return loss2
#end

# Modules for the definition of the norms for
# the observation and prior model
class Model_WeightedL2Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL2Norm, self).__init__()
    #end
    
    def forward(self, x, w, eps = 0.):
        
        loss_ = torch.nansum( x**2, dim = 3)
        loss_ = torch.nansum( loss_, dim = 2)
        loss_ = torch.nansum( loss_, dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )
        
        return loss_
    #end
#end

class Model_WeightedL1Norm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
    #end
    
    def forward(self,x,w,eps):
        
        loss_ = torch.nansum( torch.sqrt( eps**2 + x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )

        return loss_
    #end
#end

class Model_WeightedLorenzNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedLorenzNorm, self).__init__()
    #end
    
    def forward(self,x,w,eps):
        
        loss_ = torch.nansum( torch.log( 1. + eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )
        
        return loss_
    #end
#end

class Model_WeightedGMcLNorm(torch.nn.Module):
    def __init__(self):
        super(Model_WeightedL1Norm, self).__init__()
    #end
    
    def forward(self,x,w,eps):
        
        loss_ = torch.nansum( 1.0 - torch.exp( - eps**2 * x**2 ) , dim = 3)
        loss_ = torch.nansum( loss_ , dim = 2)
        loss_ = torch.nansum( loss_ , dim = 0)
        loss_ = torch.nansum( loss_ * w )
        loss_ = loss_ / (torch.sum(~torch.isnan(x)) / x.shape[1] )
        
        return loss_
    #end
#end

def compute_WeightedL2Norm1D(x2,w):
    loss_ = torch.nansum(x2**2 , dim = 2)
    loss_ = torch.nansum( loss_ , dim = 0)
    loss_ = torch.nansum( loss_ * w )
    loss_ = loss_ / (torch.sum(~torch.isnan(x2)) / x2.shape[1] )
    
    return loss_
#end


# Gradient-based minimization using a LSTM using a (sub)gradient as inputs
class ConvLSTM2d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3, stochastic=False):
        super(ConvLSTM2d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv2d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        self.stochastic = stochastic
        #self.correlate_noise = CorrelateNoise(input_size, 10)
        #self.regularize_variance = RegularizeVariance(input_size, 10)
    #end
    
    def forward(self, input_, prev_state):
        
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        if self.stochastic == True:
            z = torch.randn(input_.shape).to(DEVICE)
            z = self.correlate_noise(z)
            z = (z-torch.mean(z))/torch.std(z)
            #z = torch.mul(self.regularize_variance(z),self.correlate_noise(z))
        #end
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(DEVICE),
                torch.autograd.Variable(torch.zeros(state_size)).to(DEVICE)
            )
        #end
        
        # prev_state has two components
        prev_hidden, prev_cell = prev_state
        
        # data size is [batch, channel, height, width]
        if self.stochastic == False:
            stacked_inputs = torch.cat((input_, prev_hidden), 1)
        else:
            stacked_inputs = torch.cat((torch.add(input_, z), prev_hidden), 1)
        #end
        
        gates = self.Gates(stacked_inputs)
        
        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        
        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        
        return hidden, cell
    #end
#end

class ConvLSTM1d(torch.nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size = 3):
        super(ConvLSTM1d, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.padding = int((kernel_size - 1) / 2)
        self.Gates = torch.nn.Conv1d(input_size + hidden_size, 4 * hidden_size, kernel_size = self.kernel_size, stride = 1, padding = self.padding)
        #end
    #end
    
    def forward(self, input_, prev_state):
        
        # get batch and spatial sizes
        batch_size = input_.shape[0]
        spatial_size = input_.shape[2:]
        
        # generate empty prev_state, if None is provided
        if prev_state is None:
            '''
            ATTENTION : the variable ``spatial_size`` has the value of
            24, which is not spatial but rather the temporal dimension.
            Could it be an issue?
            Is it the dimension on which the convolutional operators act,
            regardless of the fact they are 1d or 2d?
            '''
            # batch_size   : m
            # hidden_size  : dim state of LSTM (user defined)
            # spatial_size : FORMAT_SIZE
            state_size = [batch_size, self.hidden_size] + list(spatial_size)
            prev_state = (
                torch.autograd.Variable(torch.zeros(state_size)).to(DEVICE),
                torch.autograd.Variable(torch.zeros(state_size)).to(DEVICE)
            )
        #end
        
        # prev_state has two components
        # input_     : [m, N, T]
        # prev_state : ([m, dim_LSTM, T], [m, dim_LSTM, T])
        prev_hidden, prev_cell = prev_state
        
        # data size is [batch, channel, height, width]
        # stacked_inputs : [m, N + dim_LSTM, T]
        stacked_inputs = torch.cat((input_, prev_hidden), 1)
        gates = self.Gates(stacked_inputs)
        
        # chunk across channel dimension: split it to 4 samples at dimension 1
        in_gate, remember_gate, out_gate, cell_gate = gates.chunk(4, 1)
        
        # apply sigmoid non linearity
        in_gate = torch.sigmoid(in_gate)
        remember_gate = torch.sigmoid(remember_gate)
        out_gate = torch.sigmoid(out_gate)
        
        # apply tanh non linearity
        cell_gate = torch.tanh(cell_gate)
        
        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * torch.tanh(cell)
        
        # hidden : [m, dim_LSTM, T]
        # cell   : [m, dim_LSTM, T]
        return hidden, cell
    #end
#end

class model_GradUpdateLSTM(torch.nn.Module):
    
    def __init__(self, ShapeData, periodicBnd = False, DimLSTM = 0, rateDropout = 0., stochastic = False):
        super(model_GradUpdateLSTM, self).__init__()
        
        with torch.no_grad():
            
            self.shape = ShapeData
            if DimLSTM == 0:
                self.dim_state = 5 * self.shape[0]
            else:
                self.dim_state = DimLSTM
            #end
            
            self.PeriodicBnd = periodicBnd
            if( (self.PeriodicBnd == True) & (len(self.shape) == 2) ):
                print('No periodic boundary available for FxTime (eg, L63) tensors. Forced to False')
                self.PeriodicBnd = False
            #end
        #end
        
        self.convLayer = self._make_ConvGrad()
        K = torch.Tensor([0.1]).view(1,1,1,1)
        self.convLayer.weight = torch.nn.Parameter(K)
        
        self.dropout = torch.nn.Dropout(rateDropout)
        self.stochastic = stochastic
        
        if len(self.shape) == 2: ## 1D Data
            self.lstm = ConvLSTM1d(self.shape[0], self.dim_state, 3)
        elif len(self.shape) == 3: ## 2D Data
            self.lstm = ConvLSTM2d(self.shape[0], self.dim_state, 3, stochastic = self.stochastic)
        #end
    #end
    
    def _make_ConvGrad(self):
        
        layers = []
        if len(self.shape) == 2: ## 1D Data
            layers.append(torch.nn.Conv1d(self.dim_state, self.shape[0], 1, padding = 0, bias = False))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(torch.nn.Conv2d(self.dim_state, self.shape[0], (1,1), padding = 0, bias = False))
        #end
        
        return torch.nn.Sequential(*layers)
    #end
    
    def _make_LSTMGrad(self):
        
        layers = []
        if len(self.shape) == 2: ## 1D Data
            layers.append(ConvLSTM1d(self.shape[0], self.dim_state, 3))
        elif len(self.shape) == 3: ## 2D Data
            layers.append(ConvLSTM2d(self.shape[0], self.dim_state, 3, stochastic = self.stochastic))
        #end
        
        return torch.nn.Sequential(*layers)
    #end
    
    def forward(self, hidden, cell, grad, gradnorm = 1.0):
        
        # compute gradient
        # hidden : [m, 2, T]
        # cell   : [m, 2, T]
        # grad   : [m, N, T]
        grad = grad / gradnorm
        grad = self.dropout( grad )
        
        if self.PeriodicBnd == True :
            dB = 7
            grad_ = torch.cat( (grad[:,:,grad.size(2)-dB:,:], grad, grad[:,:,0:dB,:]) ,dim=2 )
            if hidden is None:
                hidden_,cell_ = self.lstm(grad_, None)
            else:
                hidden_ = torch.cat( (hidden[:,:,grad.size(2)-dB:,:], hidden, hidden[:,:,0:dB,:]), dim=2 )
                cell_ = torch.cat( (cell[:,:,grad.size(2)-dB:,:], cell, cell[:,:,0:dB,:]), dim=2 )
                hidden_,cell_ = self.lstm(grad_, [hidden_, cell_])
            #end
            
            hidden = hidden_[:,:,dB:grad.size(2)+dB,:]
            cell = cell_[:,:,dB:grad.size(2)+dB,:]
        else:
            if hidden is None:
                
                # grad   : [m, N, T]
                # hidden : []
                # cell   : []
                hidden, cell = self.lstm(grad, None)
            else:
                hidden, cell = self.lstm(grad, [hidden, cell])
            #end
        #end
        
        grad = self.dropout( hidden ) # grad : [m, dim_LSTM, T]
        grad = self.convLayer( grad ) # grad : [m, N, T]
        
        return grad, hidden, cell
    #end
#end


# New module for the definition/computation of the variational cost
# TIP
class Model_Var_Cost(nn.Module):
    
    def __init__(self, m_NormObs, m_NormPhi, ShapeData,
                 dim_obs = 1, dim_obs_channel = 0, dim_state = 0,
                 learnable_params = False,
                 alphaObs = 1., alphaReg = 1.):
        super(Model_Var_Cost, self).__init__()
        
        self.dim_obs_channel = dim_obs_channel
        self.dim_obs = dim_obs
        
        if dim_state > 0 :
            self.dim_state = dim_state
        else:
            self.dim_state = ShapeData[0]
        #end
        
        self.normObs   = m_NormObs
        self.normPrior = m_NormPhi
        
        # parameters for variational cost
        if learnable_params:
            
            self.alphaObs = torch.nn.Parameter(torch.Tensor(1. * np.ones((self.dim_obs,1)))).to(DEVICE)
            self.alphaReg = torch.nn.Parameter(torch.Tensor([1.])).to(DEVICE)
            
            if self.dim_obs_channel[0] == 0 :
                
                self.WObs = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,ShapeData[0])))).to(DEVICE)
                self.dim_obs_channel = ShapeData[0] * np.ones((self.dim_obs,))
            else:
                
                self.WObs = torch.nn.Parameter(torch.Tensor(np.ones((self.dim_obs,np.max(self.dim_obs_channel))))).to(DEVICE)
            #end
            
            self.WReg = torch.nn.Parameter(torch.Tensor(np.ones(self.dim_state,)))
            self.epsObs = torch.nn.Parameter(0.1 * torch.Tensor(np.ones((self.dim_obs,))))
            self.epsReg = torch.nn.Parameter(torch.Tensor([0.1]))
            
        else:
            
            self.alphaObs = torch.Tensor([alphaObs]).to(DEVICE)
            self.alphaReg = torch.Tensor([alphaReg]).to(DEVICE)
            
            if self.dim_obs_channel[0] == 0:
                
                self.WObs = torch.Tensor(np.ones((self.dim_obs,ShapeData[0])))
                self.dim_obs_channel = ShapeData[0] * np.ones((self.dim_obs,))
            else:
                
                self.WObs = torch.Tensor(np.ones((self.dim_obs, np.max(self.dim_obs_channel))))
            #end
            
            self.WReg = torch.Tensor(np.ones(self.dim_state,))
            self.epsObs = 0.1 * torch.Tensor(np.ones((self.dim_obs,)))
            self.epsReg = torch.Tensor([0.1])
        #end
    #end
    
    def forward(self, data_fidelty, regularization):
        
        loss = self.alphaReg.pow(2) * self.normPrior(regularization)
        
        if self.dim_obs == 1:
            loss += self.alphaObs[0].pow(2) * self.normObs(data_fidelty)
        else:
            for kk in range(0, self.dim_obs):
                loss += self.alphaObs[kk,0].pow(2) * self.normObs(data_fidelty[kk])
            #end
        #end
        
        return loss
    #end
#end

# 4DVarNN Solver class using automatic differentiation for the computation of gradient of the variational cost
# input modules: operator phi_r, gradient-based update model m_Grad
# modules for the definition of the norm of the observation and prior terms given as input parameters 
# (default norm (None) refers to the L2 norm)
# updated inner modles to account for the variational model module
class Solver_Grad_4DVarNN(nn.Module):
    
    def __init__(self,
                 prior,
                 mod_H,
                 m_Grad,
                 m_NormObs,
                 m_NormPhi,
                 ShapeData,
                 n_iter_grad,
                 alphaObs = 1.,
                 alphaReg = 1.,
                 stochastic = False,
                 varcost_learnable_params = False):
        
        super(Solver_Grad_4DVarNN, self).__init__()
        
        self.Phi = prior
        
        if m_NormObs == None:
            m_NormObs =  Model_WeightedL2Norm()
        if m_NormPhi == None:    
            m_NormPhi = Model_WeightedL2Norm()
        #end
        
        self.model_H = mod_H
        self.model_Grad = m_Grad
        self.model_VarCost = Model_Var_Cost(m_NormObs, 
                                            m_NormPhi, 
                                            ShapeData, 
                                            mod_H.dim_obs, 
                                            mod_H.dim_obs_channel,
                                            alphaObs = alphaObs,
                                            alphaReg = alphaReg,
                                            learnable_params = varcost_learnable_params)
        
        self.stochastic = stochastic
        
        self.var_cost_values = list()
        
        with torch.no_grad():
            self.n_grad = int(n_iter_grad)
        #end
    #end
    
    def forward(self, x, yobs, mask):
        
        return self.solve(x_0 = x, obs = yobs, mask = mask)
    #end
    
    def solve(self, x_0, obs, mask):
        
        x_k = torch.mul(x_0, 1.)
        hidden = None
        cell = None 
        normgrad_ = 0.
        
        var_cost_values_tmp = np.zeros(self.n_grad)
        
        for _iter in range(self.n_grad):
            
            x_k_plus_1, hidden, cell, normgrad_, vvar_cost = self.solver_step(x_k, obs, mask, hidden, cell, normgrad_)
            x_k = torch.mul(x_k_plus_1, 1.)
            var_cost_values_tmp[_iter] = vvar_cost
        #end
        
        self.var_cost_values.append(var_cost_values_tmp)
        return x_k_plus_1, hidden, cell, normgrad_
    #end
    
    def solver_step(self, x_k, obs, mask, hidden, cell, normgrad = 0.):
        
        var_cost, var_cost_grad = self.var_cost(x_k, obs, mask)
        
        if normgrad == 0. :
            normgrad_= torch.sqrt(torch.mean(var_cost_grad**2 + 0.))
        else:
            normgrad_= normgrad
        #end
        
        if self.Phi.__class__ is list or self.Phi.__class__ is nn.ModuleList:
            
            gradients = list()
            hiddens   = list()
            cells     = list()
            
            for i in range(self.Phi.__len__()):
                
                current_var_cost_grad = var_cost_grad[:,:,:,:,i]
                try:
                    current_hidden = hidden[:,:,:,:,i]
                    current_cell   = cell[:,:,:,:,i]
                except:
                    current_hidden = None
                    current_cell   = None
                #end
                
                g, h, c = self.model_Grad(current_hidden, current_cell, current_var_cost_grad, normgrad_)
                gradients.append(g)
                hiddens.append(h)
                cells.append(c)
            #end
            
            grad   = torch.stack(gradients, dim = -1)
            hidden = torch.stack(hiddens, dim = -1)
            try:
                cells  = torch.stack(cells, dim = -1)
            except:
                pass
            #end
            
            # vc_grad_mwind = var_cost_grad[:,:,:,:,0]
            # vc_grad_theta = var_cost_grad[:,:,:,:,1]
            
            # try:
            #     hidden_mwind = hidden[:,:,:,:,0]; cell_mwind = cell[:,:,:,:,0]
            #     hidden_theta = hidden[:,:,:,:,1]; cell_theta = cell[:,:,:,:,1]
            # except:
            #     hidden_mwind = None; cell_mwind = None
            #     hidden_theta = None; cell_theta = None
            # #end
            
            # grad_mwind, hidden_mwind, cell_mwind = self.model_Grad(hidden_mwind, cell_mwind, vc_grad_mwind, normgrad_)
            # grad_theta, hidden_theta, cell_theta = self.model_Grad(hidden_theta, cell_theta, vc_grad_theta, normgrad_)
            
            # grad = torch.stack([grad_mwind, grad_theta], dim = -1)
            # hidden = torch.stack([hidden_mwind, hidden_theta], dim = -1)
            
            # try:
            #     cell = torch.stack([cell_mwind, cell_theta], dim = -1)
            # except:
            #     pass
            # #end
        else:
            grad, hidden, cell = self.model_Grad(hidden, cell, var_cost_grad, normgrad_)
        #end
        
        grad *= 1./ self.n_grad
        x_k_plus_1 = x_k - grad
        
        return x_k_plus_1, hidden, cell, normgrad_, var_cost
    #end
    
    def var_cost(self, x, yobs, mask):
        
        # # qui poi ci metto self.model_H([x, x_situ], [yobs, ysitu], [mask, mask_situ])
        # if self.model_H.dim_obs == 1:
        #     data_fidelty = self.model_H(x, yobs, mask[2])
        # elif self.model_H.dim_obs > 1:
        #     x_lr = x[:,:24,:,:];    x_an = x[:,24:48,:,:]
        #     y_lr = yobs[:,:24,:,:]; y_an = yobs[:,24:48,:,:]
        #     data_fidelty = self.model_H([x_lr, x_an], [y_lr, y_an], mask)
        # #end
        data_fidelty = self.model_H(x, yobs, mask)
        
        if self.Phi.__class__ is list or self.Phi.__class__ is nn.ModuleList:
            
            regularization = torch.zeros(x.shape)
            for i, phi in enumerate(self.Phi):
                regularization[:,:,:,:,i] = x[:,:,:,:,i] - phi(x[:,:,:,:,i])
            #end
        else:
            regularization = x - self.Phi(x)
        #end
        
        var_cost = self.model_VarCost(data_fidelty, regularization)
        var_cost_grad = torch.autograd.grad(var_cost, x, create_graph = True)[0]
        
        return var_cost, var_cost_grad
    #end
#end

