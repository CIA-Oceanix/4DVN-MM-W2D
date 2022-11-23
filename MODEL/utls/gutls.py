
import os
import pickle
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch


def plot_all(pdata, ppool, preco, plcvs, path_eval, model):
    
    fig = img_data_grad_distr(pdata)
    # fig.savefig(os.path.join(path_eval, f'{model}-data-grads.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-data-grads.fig'), 'wb'))
    
    fig = img_distribution(pdata)
    # fig.savefig(os.path.join(path_eval, f'{model}-data-hist.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-data-hist.fig'), 'wb'))
    
    fig = img_data_reco_mse(pdata, ppool, 'Interpolation')
    # fig.savefig(os.path.join(path_eval, f'{model}-data-bline-mse.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-data-bline-mse.fig'), 'wb'))
    
    fig = img_data_reco_mse(pdata, preco, 'Model')
    # fig.savefig(os.path.join(path_eval, f'{model}-data-model-mse.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-data-model-mse.fig'), 'wb'))
    
    fig = img_baseline_model(pdata, preco, ppool)
    # fig.savefig(os.path.join(path_eval, f'{model}-bline-model.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-bline-model.fig'), 'wb'))
    
    fig = img_grads_mse(pdata, preco)
    # fig.savefig(os.path.join(path_eval, f'{model}-grads-mse.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-grads-mse.fig'), 'wb'))
    
    # fig = hist_pdf(cp_data, cp_reco, cp_pool)
    # # fig.savefig(os.path.join(path_eval, f'{model}-hist.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    # pickle.dump(fig, open(os.path.join(path_eval, f'{model}-hist.fig'), 'wb'))
    
    fig = learning_curves(plcvs)
    # fig.savefig(os.path.join(path_eval, f'{model}-lcvs.png'), format = 'png', dpi = 300, bbox_inches = 'tight')
    pickle.dump(fig, open(os.path.join(path_eval, f'{model}-lcvs.fig'), 'wb'))
#end

def img_data_grad_distr(_data, bidx = 0, tidx = 0):
    
    data = _data.clone().numpy()
    gradients = np.gradient(data, axis = (3,2))
    grad = np.sqrt( np.power(gradients[0], 2) + np.power(gradients[1], 2) )
    
    fig, ax = plt.subplots(1,2)
    ax0 = ax[0].imshow(data[bidx,tidx,:,:], cmap = 'Blues')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax0, cax = cax, orientation = 'vertical')
    ax[0].set_title('Data')
    
    ax1 = ax[1].imshow(grad[bidx,tidx], cmap = 'nipy_spectral', vmax = grad[bidx,tidx,:,:].max(), vmin = grad[bidx,tidx,:,:].min())
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax1, cax = cax, orientation = 'vertical')
    ax[1].set_title('Gradients norm')
    
    for i in range(2): ax[i].set_xticks([]); ax[i].set_yticks([])
    fig.tight_layout()
    # plt.show()
    
    return fig
#end

def img_distribution(_data):
    
    data = _data.clone()
    fig, ax = plt.subplots()
    ax.hist(data.flatten().numpy(), bins = 50, facecolor = 'tab:blue', alpha = 0.5)
    ax.set_title('Wind values distribution')
    ax.set_xlabel('Wind values')
    # plt.show()
    return fig
#end

def img_data_reco_mse(_data, _reco, label, bidx = 0, tidx = 0):
    
    data = _data.clone(); reco = _reco.clone()
    errors = (data - reco).pow(2).div(data.std())
    
    wmax = np.max([data[bidx, tidx].max(), reco[bidx, tidx].max()])
    wmin = np.min([data[bidx, tidx].min(), reco[bidx, tidx].min()])
    
    fig, ax = plt.subplots(1,4)
    ax0 = ax[0].imshow(data[bidx, tidx], cmap = 'Blues', vmax = wmax, vmin = wmin)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax0, cax = cax, orientation = 'vertical')
    ax[0].set_title(r'Data $u$')
    
    ax1 = ax[1].imshow(reco[bidx, tidx], cmap = 'Blues', vmax = wmax, vmin = wmin)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax1, cax = cax, orientation = 'vertical')
    ax[1].set_title(label)
    
    ax2 = ax[2].imshow(errors[bidx, tidx], cmap = 'nipy_spectral')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax2, cax = cax, orientation = 'vertical')
    ax[2].set_title(r'MSE')
    
    abs_difference = (data - reco)
    
    ax3 = ax[3].imshow(abs_difference[bidx, tidx], cmap = 'Blues')
    divider = make_axes_locatable(ax[3])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax3, cax = cax, orientation = 'vertical')
    ax[3].set_title(r'Absolute difference')
    
    for i in range(4): ax[i].set_xticks([]); ax[i].set_yticks([])    
    fig.tight_layout()
    # plt.show()
    
    return fig
#end

def img_baseline_model(_data, _reco, _pool, bidx = 0, tidx = 0):
    
    data = _data.clone(); reco = _reco.clone(); pool = _pool.clone()
    wmax = np.max([data[bidx, tidx].max(), reco[bidx, tidx].max()])
    wmin = np.min([data[bidx, tidx].min(), reco[bidx, tidx].min()])
    
    fig, ax = plt.subplots(1,3)
    ax1 = ax[0].imshow(data[bidx, tidx,:,:], vmin = wmin, vmax = wmax, cmap = 'Blues')
    ax[0].set_xticks([]); ax[0].set_yticks([])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax1, cax = cax, orientation = 'vertical')
    ax[0].set_title('Data')
    
    ax2 = ax[1].imshow(pool[bidx, tidx,:,:], vmin = wmin, vmax = wmax, cmap = 'Blues')
    ax[1].set_xticks([]); ax[1].set_yticks([])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax2, cax = cax, orientation = 'vertical')
    ax[1].set_title('Interpolation')
    
    ax5 = ax[2].imshow(reco[bidx, tidx,:,:], vmin = wmin, vmax = wmax, cmap = 'Blues')
    ax[2].set_xticks([]); ax[2].set_yticks([])
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax5, cax = cax, orientation = 'vertical')
    ax[2].set_title('Model')
    
    fig.tight_layout()
    # plt.show()
    return fig
#end

def img_grads_mse(_data, _reco, bidx = 0, tidx = 0):
    
    data = _data.clone(); reco = _reco.clone()
    
    g_data_x, g_data_y = torch.gradient(data, dim = (-1,-2))
    g_reco_x, g_reco_y = torch.gradient(reco, dim = (-1,-2))
    
    grad_data = torch.sqrt( (g_data_x.pow(2) + g_data_y.pow(2)) )
    grad_reco = torch.sqrt( (g_reco_x.pow(2) + g_reco_y.pow(2)) )
    
    error_grad = (grad_data - grad_reco).pow(2)
    
    fig, ax = plt.subplots(1,3, figsize = (10,5))
    ax1 = ax[0].imshow(grad_data[bidx, tidx,:,:], cmap = 'Blues', vmax = grad_data[bidx,tidx,:,:].max(), vmin = grad_data[bidx,tidx,:,:].min())
    ax[0].set_xticks([]); ax[0].set_yticks([])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax1, cax = cax, orientation = 'vertical')
    ax[0].set_title(r'Norm gradient data')
    
    ax2 = ax[1].imshow(grad_reco[bidx, tidx,:,:], cmap = 'Blues', vmax = grad_reco[bidx,tidx,:,:].max(), vmin = grad_reco[bidx,tidx,:,:].min())
    ax[1].set_xticks([]); ax[1].set_yticks([])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax2, cax = cax, orientation = 'vertical')
    ax[1].set_title(r'Norm gradient reco')
    
    ax3 = ax[2].imshow(error_grad[bidx, tidx,:,:], cmap = 'nipy_spectral')
    ax[2].set_xticks([]); ax[2].set_yticks([])
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size = '5%', pad = 0.05)
    fig.colorbar(ax3, cax = cax, orientation = 'vertical')
    ax[2].set_title(r'Norm gradient SE')
    
    # plt.show()
    return fig
#end

def hist_pdf(_data, _reco, _pool = None, nbins = 50, bidx = 0, tidx = 0):
    
    data = _data.clone(); reco = _reco.clone()
    
    data_flat = data.flatten().numpy()
    reco_flat = reco.flatten().numpy()
    if _pool is not None:
        pool = _pool.clone()
        pool_flat = pool.flatten().numpy()
    #end
    
    fig, ax = plt.subplots()
    ax.hist(data_flat, bins = nbins, facecolor = 'tab:blue', alpha = 0.5, label = 'Data')
    ax.hist(reco_flat, bins = nbins, facecolor = 'tab:orange', alpha = 0.5, label = 'Model')
    if _pool is not None:
        ax.hist(pool_flat, bins = nbins, facecolor = 'tab:green', alpha = 0.5, label = 'Baseline')
    #end
    ax.legend()
    ax.set_xlabel('Wind - Central patch')
    # plt.show()
    
    return fig
#end

def learning_curves(lcvs):
    
    fig, ax = plt.subplots()
    ax.plot(lcvs['train'], c = 'g', lw = 2, alpha = 0.75, label = 'Train')
    ax.plot(lcvs['val'],   c = 'r', lw = 2, alpha = 0.75, label = 'Validation')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('MSE')
    ax.grid()
    ax.legend()
    # plt.show()
    
    return fig
#end