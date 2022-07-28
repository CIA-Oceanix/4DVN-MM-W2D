
import torch
from torch import nn
import numpy as np
import warnings


#------------------------------------------------------------------------------
# M O D E L S
#------------------------------------------------------------------------------

# Parameters init

def xavier_weights_initialization(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
    #end
#end



# AE --------------------------------------------------------------------------

class Encoder(nn.Module):
    
    def __init__(self, encoder):
        super(Encoder, self).__init__()
        
        self.encoder = encoder
        self.apply(xavier_weights_initialization)
        #end
    #end
    
    def forward(self, x):
        
        return self.encoder(x)
    #end
#end

class Decoder(nn.Module):
    
    def __init__(self, decoder):
        super(Decoder, self).__init__()
        
        self.decoder = decoder
        self.apply(xavier_weights_initialization)
        #end
    #end
    
    def forward(self, z):
        
        return self.decoder(z)
    #end
#end

class AutoEncoder(nn.Module):
    
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()
        
        self.encoder = Encoder(encoder = encoder)
        self.decoder = Decoder(decoder = decoder)
        
    #end
    
    def forward(self, x):
        
        return self.decoder(self.encoder(x))
    #end
#end


# LSTM ------------------------------------------------------------------------

class LSTMSolver(nn.Module):
    
    def __init__(self, lstm_solver, linear_map):
        super(LSTMSolver, self).__init__()
        
        self.lstm_solver = lstm_solver
        self.linear_map  = linear_map
    #end
    
    def forward(self, grad):
        
        lstm_out, (hidden_state, cell_state) = self.lstm_solver(grad)
        return self.linear_map(lstm_out)
    #end
#end


# VAE -------------------------------------------------------------------------

class ProbabilisticEncoder(nn.Module):
    
    def __init__(self, encoder, enc_mean, enc_logvar, modality):
        super(ProbabilisticEncoder, self).__init__()
        
        self.encoder    = encoder
        self.enc_mean   = enc_mean
        self.enc_logvar = enc_logvar
        self.modality   = modality
        
        self.apply(xavier_weights_initialization)
    #end
    
    def baptize(self):
        
        self.name = 'Encoder {}'.format(self.modality)
    #end
    
    def reparametrize(self, enc_mean, enc_logvar):
        
        if enc_mean.__class__ is not torch.Tensor:
            enc_mean = torch.Tensor(enc_mean).type(torch.float32)
        if enc_logvar.__class__ is not torch.Tensor:
            enc_logvar = torch.Tensor(enc_logvar).type(torch.float32)
        #end
        
        z = torch.distributions.Normal(enc_mean, torch.exp(0.5 * enc_logvar)).rsample()
        return z
    #end
    
    def forward(self, data):
        
        hidden     = self.encoder(data)
        enc_mean   = self.enc_mean(hidden)
        enc_logvar = self.enc_logvar(hidden)
        
        z = self.reparametrize(enc_mean, enc_logvar)
        return z, enc_mean, enc_logvar
    #end
#end

class ProbabilisticDecoder(nn.Module):
    
    def __init__(self, decoder, modality):
        super(ProbabilisticDecoder, self).__init__()
        
        self.decoder    = decoder
        self.modality   = modality
        
        self.apply(xavier_weights_initialization)
    #end
    
    def baptize(self):
        
        self.name = 'Decoder {}'.format(self.modality)
    #end
    
    def model_sample(self, dec_mean, dec_logvar = torch.Tensor([0.])):
        
        loc       = dec_mean
        scale     = torch.exp(0.5 * dec_logvar)
        posterior = torch.distributions.Normal(loc, scale)
        reco      = posterior.rsample()
        return reco
    #end
    
    def forward(self, latent):
        
        reco = self.decoder(latent)
        return reco
    #end
#end


# CODE FUNCTION G : Z x Z -> Z ------------------------------------------------

def CodeFunction(x_code, y_code, operation = 'average'):
    
    if operation == 'average':
        if torch.all(x_code == 0.) or torch.all(y_code == 0.):
            factor = 1.
        else:
            factor = 2.
        #end
        
        code = torch.add(x_code, y_code).div(factor)
    
    if operation == 'concatenation':
        
        code = torch.cat( (x_code, y_code), axis = 1 )
    #end
    
    return code
#end


# WIND MAP f : Z -> R ---------------------------------------------------------


class RegressionNetwork(nn.Module):
    
    def __init__(self, net):
        super(RegressionNetwork, self).__init__()
        
        self.net = net
        self.apply(xavier_weights_initialization)
    #end
    
    def forward(self, x):
        
        return self.net(x)
    #end
#end



# -----------------------------------------------------------------------------
# L O S S E S
# -----------------------------------------------------------------------------

# NORM LOSS AND MISSING VALUES

def Norm2Loss(item, mask = None, weight = 1, p = 2, rmse = False, divide = True,
             format_size = np.int32(1), return_nitems = False):
    r'''
    DEPRECATED AND NO MORE MAINTAINED. SOON TO DELETE.
    
    REFER TO ``NormLoss`` INSTEAD
    '''
    
    if mask is None:
        mask = torch.ones_like(item)
    #end
    
    if p == 2:
        argument = (item.mul(mask)).pow(p)
    if p == 1:
        argument = (item.mul(mask)).abs()
    #end
    
    loss = argument.sum(dim = -1)
    
    no_items = False
    nitems = 1
    
    if mask.shape.__len__() == 0:
        no_items = True
        nitems = 1.
    else:
        if mask.sum() == 0.:
            no_items = True
            nitems = 1.
        else:
            no_items = False
            num_features = mask.shape[-1] / format_size
            nitems = mask.sum() / num_features
        #end
    #end
    
    if item.size().__len__() == 3:
        loss = loss.sum(dim = 0).sum()
    elif item.size().__len__() <= 2:
        loss = loss.sum(dim = -1)
    #end
    
    if divide:
        loss = loss.div(nitems)
    #end
    
    if rmse:
        loss = loss.pow(1 / p)
    #end
    
    if no_items == True:
        nitems = 0
    #end
    
    if return_nitems:
        return loss * weight, nitems
    else:
        return loss * weight
    #end

    # np.sqrt( (u - u_reco).mul(muv).pow(2).sum(dim = -1).sum(dim = 0).sum().div(nitems).item() )
#end


def NormLoss(item, mask, rmse = False, divide = True, dformat = 'mtn', return_nitems = False):
    r'''Computes the norm loss of the input ``item``, a generic tensor :math:`v`,
    which can be given by the difference between ground-truth data and reconstructions
    or predictions, i.e. :math:`\mathrm{item} = (v - v')`. The presence of missing
    data is accounted for in the following way. Assume that the data is :math:`v \in \mathbb{R}^{D^v}`,
    with :math:`D^v = \{D_1^v, \dots, D_d^v\}` a multi-index (batch size, timesteps, number of features).
    Then we mask this data batch with a mask :math:`\Omega^v \in \{0, 1\}^{D^v}`, such that
    
    .. math::
        \Omega_{itj}^v = 
        \begin{cases}
            1 \quad \text{if} \;\; v_{itj} \in \mathbb{R} \\
            0 \quad \text{if} \;\; v_{itj} \; \text{is non-numeric}
        \end{cases}
    
    The mask is used for masking missing values and to infer the number of
    legitimate present data. If the loss has a value :math:`L`, but only one item
    is responsible, i.e. there is only one legitimate value, then the loss
    "pro-capita" is :math:`L` whatsoever. But if we have a loss :math:`L` being
    caused by :math:`m` legitimate items, then the loss pro-capita is :math:`L / m`,
    which is the unbiased batch-wise loss we are interested in.
    
    The actual loss computation is then 
    
    .. math::
        L(v,v') = \sum_{t = 0}^{T} \frac{1}{\sum_{i = 1}^{m}\sum_{j = 0}^{N^v} \Omega_{itj}^v / Nv} 
          \sum_{i = 0}^{m}\sum_{j = 0}^{N^v} \Omega_{itj}^v \, ( v_{itj} - v_{itj}' )^2
    
    The normalization term accounts matches with the number legitimate data (sum of the
    elements of the mask divided by the number of features). It may give an estimate
    of the actual number of data, since the assumption is that data are row-wise 
    present or missing.
    
    In test stage it could be useful to consider the root mean squared error version of
    this loss, and it sufficies to pass the ``rmse = True`` keyword.
    
    **NOTE**: This function has been developed and tested in such a way for 
    comply for the equality between the two::
        >>> NormLoss(v, p = 2, rmse = True, divide = False)    
        >>> torch.linalg.norm(v, ord = 2, dim = -1).sum()
    
    since we are interested in a row-wise norm, to be summed over the batch items
    (indeed the rows of the tensor :math:`v`).
    
    **NOTE**: The equality between ``NormLoss`` and ``torch.nn.functional.mse_loss`` or
    ``sklearn.metrics.mean_squared_error`` should be expected only if the tensor ``item``
    is a 2-dimensional element. It is due to the fact that this method is designed to
    compute the mean squared error of time series, but the temporal dimension is not 
    accounted for in the average. The mean error is cumulated for each time step. It is
    thought that dividing also for the time series length ends up underestimating the 
    performance metric. In the case of wind speed, however, there is no difference.
    
    Parameters
    ----------
    item : ``torch.Tensor``
        The tensor which one wants the norm of.
    mask : ``torch.Tensor``
        The mask :math:`\Omega^v`. The default is None, in this case the mask is 
        assumed to be a tensor of ones, like if there were no missing values, and
        each item contributes to the loss.
    rmse : ``bool``
        Whether to compute the square root of the norm or not. The default is False.
    divide : ``bool``
        Whether to divide the loss for the number of the present items. The default is True.
    dformat : ``str``. Defaults to ``'mtn'``
        The format of the data passed. ``m`` in the batch dimension, ``t`` is the time
        dimension and ``n`` is the features dimension. The preferable choice is 
        ``'mtn'``, meaning that the data passed are in the following format : 
        ``(batch_size, time_series_length, num_features)``. In this way the loss
        reductions are done in such a way to comply with the equation above.
    return_nitems : ``bool``
        Whether to return the number of effective items in the given batch. Default
        is ``False``.
        
    Returns
    -------
    loss : ``torch.Tensor``
        Weighted loss computed, if ``return_nitems`` is ``False``.
    loss, nitems : ``tuple`` of ``(torch.Tensor, int)``
        The weighted loss and the number of effective items in the batch.

    '''
    if item.__class__ is not torch.Tensor:
        item = torch.Tensor(item)
    #end
    
    if item.shape.__len__() == 1:
        item = item.reshape(-1,1)
    #end
    
    if mask is None:
        mask = torch.ones_like(item)
    #end
    
    argument = (item.mul(mask)).pow(2)
    
    # FIRST SUMMATION : ON FEATURES !!!
    if dformat == 'mtn':
        loss = argument.sum(dim = -1)
    elif dformat == 'mnt':
        loss = argument.sum(dim = 1)
    #end
    
    no_items = False
    nitems = 1
    
    if mask.sum() == 0.:
        no_items = True
        nitems = 1.
    else:
        no_items = False
        num_features = mask.shape[-1]
        nitems = mask.sum() / num_features
    #end
    
    loss = loss.sum(dim = 0).sum() # SUMMATION OVER BATCH FIRST AND OVER TIME THEN
    
    if divide:
        loss = loss.div(nitems)
    #end
    
    if rmse:
        loss = loss.sqrt()
    #end
    
    if no_items == True:
        nitems = 0
    #end
    
    if return_nitems:
        return loss, nitems
    else:
        return loss
    #end
#end

class L2NormLoss(nn.Module):
    r'''
    Wraps the ``NormLoss`` function, makes it an instantiable class, like
    the default ``torch.nn`` modules for loss computation, e.g. ``torch.nn.MSELoss`` etc.
    '''
    
    def __init__(self, dformat = 'mnt', rmse = False, divide = True, return_nitems = False):
        r'''
        SEE ABOVE, FEW CHANGES
        '''
        super(L2NormLoss, self).__init__()
        
        self.dformat = dformat
        self.rmse = rmse
        self.divide = divide
        self.return_nitems = return_nitems
    #end
    
    def forward(self, item, mask):
        
        if mask is None: 
            mask = torch.ones_like(item)
        #end
        
        return NormLoss(item, mask, rmse = self.rmse, divide = self.divide,
                        dformat = self.dformat, return_nitems = self.return_nitems)
    #end
#end


# VARIATIONAL COST (DATA ASSIMILATION) ----------------------------------------

def VariationalCost(x, y, mask_y, dynamical_prior, weight_data = 1, weight_regularization = 1):
    r'''
    
    DEPRECATED
    
    USED THE VARIATIONAL COST IN THE 4DVARCORE SOFTWARE PACKAGE
    
    
    Computes the variational cost :math:`U_{\Phi}(x, y; \Omega)` used in variational data
    assimilation problems. Call :math:`x` the true system state and :math:`y`
    the observations. The dynamical prior :math:`\Phi` has the role of 
    modelling the physical dynamics of the system, then the variational cost
    restricted to the observation domain :math:`\Omega` can be expressed as 
    
        .. math::
            U_{\Phi}(x,y; \Omega) = \| x - y \|^2_{\Omega} + \| x - \Phi(x) \|^2
    
    This quantity is minimized to find the optimal solution :math:`x`.
    
    **Note** that the variational cost is not a MSE loss. Recall that the function
    ``NormLoss`` is designed in such a way to give the same output as the function
    ``torch.linalg.norm``, but the former should be multiplied by the number of
    legitimate items in the argument, given by
    
        .. math::
            \text{items} = \frac{1}{N_v} \sum_{i = 1}^{m} \sum_{j = 1}^{N_v} \Omega_{ij}^{v}
    
    This is done automatically once the keyword ``divide = False`` is given 
    when calling the norm-loss function.
    
    Parameters
    ----------
    x : ``torch.Tensor``
        True systems states.
    y : ``torch.Tensor``
        Observations.
    mask_y : ``torch.Tensor``
        The mask which specifies the sampling rates and/or windows of the 
        domain :math:`\Omega`.
    dynamical_prior : Callable
        Could it be a ``torch.nn.Module`` subclass
        or any homomorphic function that maps the states :math:`x` to the 
        same quantity evaluated one step ahead.
    weight_data : ``float``
        Weight on the first term, the data fidelty. Default is 1.
    weight_regularization : ``float``
        Weight on the second term, the regularization. Default is 1.

    Returns
    -------
    var_cost : ``torch.Tensor``
        Evaluation of the variational cost :math:`U_{\Phi}(x,y; \Omega)`.
    '''
    
    data_fidelty   = NormLoss((x - y), mask = mask_y, divide = False)
    z = dynamical_prior[0](x); state_forecast = dynamical_prior[1](z)
    regularization = NormLoss((x - state_forecast), mask = None, divide = False)
    
    var_cost = data_fidelty * weight_data + regularization * weight_regularization
    return var_cost
#end


# VARIATIONAL AUTOENCODERS LOSSES

def GaussianLikelihood(data, dec_mean, dec_logvar = torch.Tensor([0.])):
    r'''
    Computes the log-likelihood of the data with respect to the posterior
    parametrized with the mean and log-variance learned by the  probabilistic
    decoder. This is the second term of the ELBO objective function for 
    fitting VAEs (as in Kingma and Welling (2014), Auto-Encoding Variational Bayes),
    mathematically stated as 
    
    .. math::
        \mathbb{E}_{z \sim q_{\phi}(z | x)} \log p_{\theta}(x | z)
    
    with :math:`\phi` the variational parameters of the posterior on the latents :math:`z` and
    :math:`\theta` the variational parameters of the likelihood :math:`p_{\theta}(x | z)`.
    
    Parameters
    ----------
    data : ``torch.Tensor``
        Original data (batch).
    loc : ``torch.Tensor``
        The learned expected value of the posterior distribution.
    logscale : ``torch.Tensor``
        The learned log-variance of the posterior distribution.

    Returns
    -------
    samples : ``torch.Tensor``
        Weighted loss computed.
    
    '''
    
    loc     = dec_mean
    scale   = torch.exp(0.5 * torch.ones_like(dec_logvar))
    p       = torch.distributions.Normal(loc, scale)
    samples = p.log_prob(data).sum(dim = 1)
    return samples
#end

def KLDivergence(enc_mean, enc_logvar, z):
    '''
    Monte-Carlo Kullback-Leibler divergence. It is the first term of the ELBO
    objective function in training VAEs (refer to Kingma and Welling (2014), Auto-Encoding
    Variational Bayes)
    
    .. math::
        D_{\mathrm{KL}}(q_{\phi}(z | x) || p(z))
    
    with :math:`q_{\phi}` the variational-approximated posterior on the latents 
    :math:`z` and :math:`p(z)` the prior on the latents.
    
    This function ignores the possibility of computing analytically the KL divergence
    for the case of Gaussian prior and posteriors, and samples the vectors :math:`z`
    from the Gaussian distribution obtained with the provided parameters.

    Parameters
    ----------
    enc_mean : ``torch.Tensor``
        The variationally learned expected value of the posterior.
    enc_logvar : ``torch.Tensor``
        The variationally learned variance of the posterior.
    z : ``torch.Tensor``
        The latent vector sampled from the variational approximation of the posterior 
        parametrized with the parameters learned from the decoder.

    Returns
    -------
    KLD : ``torch.Tensor``
        The estimation of the KL divergence.

    '''
    
    scale = torch.exp(0.5 * enc_logvar)
    loc   = enc_mean
    p     = torch.distributions.Normal(torch.zeros_like(loc), torch.ones_like(scale))
    q     = torch.distributions.Normal(loc, scale)
    KLD   = torch.sum(q.log_prob(z) - p.log_prob(z), dim = -1) 
    return KLD
#end

def AnKLDivergence(enc_mean, enc_logvar):
    r'''
    Analytical computation of the Kullback-Leibler divergence between the posterior
    on the latent variables and the prior on the same latentssd. It is the first term of the ELBO
    objective function in training VAEs (refer to Kingma and Welling (2014), Auto-Encoding
    Variational Bayes)
    
    .. math::
        -D_{\mathrm{KL}}(q_{\phi}(z | x) \, || \, p(z))
    
    with :math:`q_{\phi}` the variational-approximated posterior on the latents 
    :math:`z` and :math:`p(z)` the prior on the latents.
    
    We take advantage of the Gaussianity of the posterior and prior to perform the
    exact computation of the KL divergence integral.

    **Important**: The full expression of the loss function is 
    
    .. math::
        \mathcal{L}(\phi, \theta; x) = - D_{\mathrm{KL}}(q_{\phi}(z | x) \, || \, p(z)) + 
        \mathbb{E}_{z} \log p_{\theta}(x | z)
     
    and the analytical computation of the first term is 
    
    .. math::
        - D_{\mathrm{KL}}(q_{\phi}(z | x) \, || \, p(z)) = \frac{1}{2} \sum_{j = 1}^{J} \left( 
            1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2 \right)
    
    Hence in the routine for the compuation of this term we explicitly put the minus sign
    because in the following we will add the negative of the likelihood. So we have the 
    completely negativized loss function to minimize. This is because Pytorch optimizers
    do minimize, so we code the loss function (ELBO should be maximized) in such a way to
    be a minimum problem.

    Parameters
    ----------
    enc_mean : ``torch.Tensor``
        The variationally learned expected value of the posterior.
    enc_logvar : ``torch.Tensor``
        The variationally learned variance of the posterior.
    z : ``torch.Tensor``
        The latent vector sampled from the variational approximation of the posterior 
        parametrized with the parameters learned from the decoder.

    Returns
    -------
    KLD : ``torch.Tensor``
        The estimation of the KL divergence.

    '''
    
    KLD = -0.5 * torch.sum(1 + enc_logvar - enc_mean.pow(2) - enc_logvar.exp(), dim = 1)
    return KLD
#end

def ELBO_LossFunction(data, latents, posterior_params, likelihood_params):
    '''
    Computation of the ELBO objective function (the same reference once again).
    Once we have the KL divergence and the estimate of the expected value of 
    the log-likelihood, we can sum and average them.

    Parameters
    ----------
    data : ``torch.Tensor``
        Original data.
    latents : ``torch.Tensor``
        The latent vector sampled from the posterior.
    posterior_params : ``list`` of ``torch.Tensor``
        Variational parameters of the posterior :math:`q_{\phi}(z | x)`.
    likelihood_params : ``list`` of ``torch.Tensor``
        Variational parameters of the likelihood :math:`p_{\theta}(x | z)`.

    Returns
    -------
    ELBO_loss : ``torch.Tensor``
        The estimation of the ELBO loss function. Batch-wise averaged.

    '''
    
    KL_divergence = KLDivergence(posterior_params[0], posterior_params[1], latents)
    EV_likelihood = GaussianLikelihood(data, likelihood_params[0], likelihood_params[1])
    ELBO_loss     = torch.mean(KL_divergence) - torch.mean(EV_likelihood)
    return ELBO_loss

#end



# SCOASSE

def is_missing(item, missing_value = 'zero'):
    
    if missing_value == 'zero':
        return not torch.is_nonzero(item)
    if missing_value == 'nan':
        return not torch.isnan(item)
    #end
#end

def SumLosses(loss_list, missing_value = 'zero'):
    
    losses = list()
    for loss_item in loss_list:
        if not is_missing(loss_item, missing_value):
            losses.append(loss_item)
        #end
    #end
    
    if losses.__len__() == np.uint8(0):
        warnings.warn('In SumLosses: Empty losses list. Returning None')
        return None
    #end
    
    return torch.sum(torch.stack(losses))
#end

def AverageLoss(loss_list, missing_value = 'zero'):
    
    if loss_list.__len__() == np.uint8(0):
        warnings.warn('In AverageLoss: Empty losses list. Returning torch.Tensor([0.])')
        return torch.Tensor([0.])
    #end
    
    final_loss = torch.Tensor([0.])
    nonzeroes  = 0
    for loss_item in loss_list:
        if not is_missing(loss_item, missing_value):
            final_loss.add_(loss_item)
            nonzeroes += 1
        #end
    #end
    
    final_loss.div_(nonzeroes)
    return final_loss
#end
