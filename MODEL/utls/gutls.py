
'''
NOTE that for titling purposes, I load the CONSTANTS.pkl file in which I
previoulsy store all the PATH_DATA, PATH_MODEL ...
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch


PATH_PLOTS = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(PATH_PLOTS):
    os.mkdir(PATH_PLOTS)
#end

def plot_loss(train_loss, title = None, pformat = 'pdf'):
    
    fig, ax = plt.subplots(figsize = (5,4), dpi = 150)
    
    if type(train_loss) == np.ndarray:
        ax.plot(np.arange(train_loss.size), train_loss, 'k', alpha = 0.75, lw = 2)
        ylabel = 'Loss'
    if type(train_loss) == dict:
        
        loss_x = ax.plot(np.arange(train_loss['Loss X'].size), train_loss['Loss X'], color = 'b', alpha = 0.75, lw = 2, label = 'Loss X')
        loss_y = ax.plot(np.arange(train_loss['Loss Y'].size), train_loss['Loss Y'], color = 'orange', alpha = 0.75, lw = 2, label = 'Loss Y')
        loss_z = ax.plot(np.arange(train_loss['Loss Z'].size), train_loss['Loss Z'], color = 'r', alpha = 0.75, lw = 2, label = 'Loss Z')
        
        ax_ = ax.twinx()
        ax_.set_ylabel('Losses U', color = 'g')
        loss_u  = ax_.plot(np.arange(train_loss['Loss U'].size), train_loss['Loss U'], color = 'forestgreen', alpha = 0.75, lw = 2, label = 'Loss U')
        loss_ux = ax_.plot(np.arange(train_loss['Loss Ux'].size), train_loss['Loss Ux'], color = 'limegreen', alpha = 0.75, lw = 2, label = 'Loss Ux')
        loss_uy = ax_.plot(np.arange(train_loss['Loss Uy'].size), train_loss['Loss Uy'], color = 'darkgreen', alpha = 0.75, lw = 2, label = 'Loss Uy')
        ax_.tick_params(axis = 'y', labelcolor = 'g')
        
        losses = loss_x + loss_y + loss_z + loss_u + loss_ux + loss_uy
        labels = [l.get_label() for l in losses]
        ax.legend(losses, labels, bbox_to_anchor = (1.4, 1.05))
        # ax_.legend()
        ylabel = 'Losses'
    #end
    
    ax.set_xlabel('Epochs', fontsize = 12)
    ax.set_ylabel(ylabel, fontsize = 12)
    ax.grid(lw = 0.5, axis = 'both')
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    #end
    plt.show(fig)
#end


def plot_losses(losses, title = None, pformat = 'pdf'):
    
    fig, axes = plt.subplots(1,3, figsize = (10,3), dpi = 150)
    for loss, ax, color, norm, name in zip(losses, axes, ['r', 'g', 'b'],
                                            [r'$L^2$', r'$L^2$', r'$L^1$'],
                                            ['SAR images', 'UPA records', 'ECMWF wind']):
        
        losses = np.array(loss)[list(~np.isnan(loss))]
        lsize = losses.shape[0]
        upper_magnitude = np.power(10, np.floor(np.log10(lsize) + 1))
        support = np.zeros(np.int32(np.ceil(lsize / upper_magnitude) * upper_magnitude))
        support[list(np.arange(losses.size))] = losses
        
        lsupport = support.shape[0]
        grouping_size = np.int32(np.power(10, np.floor(np.log10(lsupport) - 2)))
        losses = support.reshape(grouping_size, -1).mean(axis = 0)
        
        ax.plot(np.arange(losses.size) ,losses, color = color, alpha = 0.75, lw = 1.5)
        ax.set_title(name, fontsize = 12)
        ax.set_xlabel('Epochs', fontsize = 12)
        ax.set_ylabel('{} Loss'.format(norm), fontsize = 12)
        ax.grid(axis = 'both', lw = 0.5)
    #end
    fig.tight_layout()
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    plt.show(fig)
#end


def plot_SAR(samples, title = None, pformat = 'pdf'):
    
    data = samples[0]['x_data'].detach(); data = data.reshape(-1, data.shape[-1])
    reco = samples[0]['x_reco'].detach(); reco = reco.reshape(-1, reco.shape[-1])
    
    nonzero_idx = torch.nonzero(data[:,0])[:3]
    if nonzero_idx.__len__() > 20:
        nonzero_idx = nonzero_idx[:20]
    #end
    
    img_dim = np.int32(np.sqrt(data[0].shape[-1]))
    
    fig, ax = plt.subplots(2, nonzero_idx.shape[0], figsize = (10, 5))
    for j, idx in enumerate(nonzero_idx):
        
        for i, who in zip(range(2), [data[idx], reco[idx]]):
            
            img = ax[i,j].imshow(who.view(img_dim, img_dim), cmap = 'jet')
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('right', size = '5%', pad = 0.05)
            plt.colorbar(img, cax = cax)
            ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
        #end
        
    #end
    ax[0,0].set_ylabel('$X_{data}$', fontsize = 14)
    ax[1,0].set_ylabel('$X_{reco}$', fontsize = 14)
    fig.tight_layout()
    
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = 'pdf', bbox_inches = 'tight')
    #end
    
    plt.show(fig)
#end


def plot_SAR_emulated(samples, howmany = 3, title = None, pformat = 'pdf'):
    
    data = samples[0]['x_data'].detach(); data = data.view(data.shape[0] * 24, data.shape[-1])
    reco = samples[0]['x_from_y'].detach(); reco = reco.view(reco.shape[0] * 24, reco.shape[-1])
    
    nonzero_idx = list(torch.nonzero(data[:,0])[:howmany**2])
    indices     = list(torch.arange(1, data.shape[0]))
    suitable_idx = [int(idx.numpy()) for idx in indices if idx not in nonzero_idx]
    suitable_idx = np.random.choice(np.array(suitable_idx), howmany**2)
    img_dim = np.int32(np.sqrt(data[0].shape[-1]))
    
    fig, ax = plt.subplots(howmany, howmany, figsize = (2 * howmany, 2 * howmany))
    
    idx = 0
    for i in range(howmany):
        for j in range(howmany):
            img = ax[i,j].imshow(reco[idx].reshape(img_dim, img_dim), cmap = 'jet')
            divider = make_axes_locatable(ax[i,j])
            cax = divider.append_axes('right', size = '5%', pad = 0.05)
            plt.colorbar(img, cax = cax)
            ax[i,j].set_xticks([]); ax[i,j].set_yticks([])
            idx += 1
        #end
    #end
    
    fig.tight_layout()
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    #end
    
    plt.show(fig)
#end


def plot_UPA(samples, title = None, pformat = 'pdf'):
    
    data = samples[0]['y_data'].detach(); data = data.reshape(-1, data.shape[-1])
    reco = samples[0]['y_reco'].detach(); reco = reco.reshape(-1, reco.shape[-1])
    
    nonzero_idx = torch.nonzero(data[:,0])[:3]
    if nonzero_idx.__len__() == 0:
        num_rows = 10
    else:
        num_rows = nonzero_idx.shape[0]
    #end
    
    fig, ax = plt.subplots(1, num_rows, sharex = True, figsize = (10, 3))
    for i, idx in enumerate(nonzero_idx):
        
        span = np.arange(data[idx].shape[1]).flatten()
        ax[i].plot(span, data[idx].flatten(), lw = 2, color = 'g', label = r'$Y_{data}$')
        ax[i].plot(span, reco[idx].flatten(), lw = 2, color = 'r', label = r'$Y_{reco}$')
        ax[i].grid(axis = 'both', lw = 0.5)
        ax[i].legend()
    #end
    ax[0].set_ylabel('SPL (dB)', fontsize = 14)
    ax[1].set_xlabel('Frequency', fontsize = 14)
    fig.tight_layout()
    
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    #end
    
    # plt.show(fig)
    plt.show()
#end


def plot_UPA_emulated(samples, howmany = 4, title = None, pformat = 'pdf'):
    
    data = samples[0]['y_data'].detach(); data = data.view(data.shape[0] * 24, data.shape[-1])
    reco = samples[0]['y_from_x'].detach(); reco = reco.view(reco.shape[0] * 24, reco.shape[-1])
    
    nonzero_idx = torch.nonzero(data[:,0])[:howmany]
    
    fig, ax = plt.subplots(howmany, 1, sharex = True, figsize = (3, 2 * howmany))
    
    for i, idx in enumerate(nonzero_idx):
        
        span = np.arange(data[idx].shape[1]).flatten()
        ax[i].plot(span, data[idx].flatten(), lw = 2, color = 'g', label = r'$Y_{data}$')
        ax[i].plot(span, reco[idx].flatten(), lw = 2, color = 'r', label = r'$Y_{reco}$')
        ax[i].grid(axis = 'both', lw = 0.5)
        ax[i].legend()
    #end
    ax[0].set_ylabel('SPL (dB)', fontsize = 14)
    ax[1].set_xlabel('Frequency', fontsize = 14)
    fig.tight_layout()
    
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    #end
    
    plt.show(fig)
#end


def plot_WS(samples, title = None, pformat = 'pdf'):
    
    data = samples[0]['u_data'].detach(); data = data.view(data.shape[0], 24, data.shape[-1])
    reco = samples[0]['u_reco'].detach(); reco = reco.view(reco.shape[0], 24, reco.shape[-1])
    
    # mismatch_from_mean = (reco - data.mean()).mean()
    
    fig, ax = plt.subplots(5, 2, sharex = True, sharey = True, figsize = (15,10))
    for k in range(10):
        
        j = k // 5
        i = k - j * 5
        span = np.arange(data.shape[1])
        ax[i,j].plot(span, data[k,:], color = 'k', lw = 2, label = r'$u_{data}$')
        ax[i,j].plot(span, reco[k,:], color = 'b', lw = 2, label = r'$u_{reco}$')
        ax[i,j].grid(axis = 'both', lw = 0.5)
        ax[i,j].legend()
    #end
    ax[2,0].set_ylabel('Wind speed [m/s]', fontsize = 14)
    ax[4,0].set_xlabel('Time [h]', fontsize = 14)
    ax[4,1].set_xlabel('Time [h]', fontsize = 14)
    fig.tight_layout()
    
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    #end
    
    # plt.show(fig)
    plt.show()
#end

def plot_WS_scatter(samples, mod_name, title = None, exclude = False, pformat = 'pdf'):
    
    data = samples[0]['u_data'].detach(); data = data.reshape(-1, data.shape[-1])
    reco = samples[0]['u_reco'].detach(); reco = reco.reshape(-1, reco.shape[-1])
    
    if exclude:
        x = samples[0]['{}_data'.format(mod_name)].detach(); x = x.view(-1, x.shape[-1])
        nonzero_idx = torch.nonzero(x[:,0])
        data = data[nonzero_idx].view(-1)
        reco = reco[nonzero_idx].view(-1)
    #end
    
    u_max = torch.max(data.max(), reco.max())
    bisector = np.linspace(0, u_max, 100)
    
    fig, ax = plt.subplots(figsize = (5, 5))
    
    ax.scatter(data, reco, alpha = 0.75)
    ax.set_xlabel('True values [m/s]', fontsize = 14)
    ax.set_ylabel('Predicted values [m/s]', fontsize = 14)
    ax.plot(bisector, bisector, color = 'k', alpha = 0.75, lw = 2)
    ax.grid(axis = 'both', lw = 0.5)
    
    if title is not None:
        fig.savefig(os.path.join(PATH_PLOTS, title), dpi = 300, format = pformat, bbox_inches = 'tight')
    #end
    
    # plt.show(fig)
    plt.show()
#end
