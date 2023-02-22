import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'utls'))
sys.path.append(os.path.join(os.getcwd(), '4dvar'))

from dotenv import load_dotenv
import datetime
import glob
import json
import argparse
from collections import namedtuple
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import CSVLogger

from pathmng import PathManager
from dlmodels import LitModel_OSSE1_WindModulus, LitModel_OSSE1_WindComponents, model_selection
from dutls import W2DSimuDataModule

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    print('Program runs using device : {}\n'.format(DEVICE))
else:
    DEVICE = torch.device('cpu')
    print('Program runs using device : {}\n'.format(DEVICE))
#end
# torch.autograd.set_detect_anomaly(True)

class SaveWeights(Callback):
    def __init__(self, path_ckpt):
        super(Callback, self).__init__()
        
        self.path_ckpt = path_ckpt
        self.last_epoch = None
    #end
    
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        
        pass
    #end
    
    def on_train_epoch_start(self, trainer, pl_module):
        
        self.epoch_start_params = os.path.join(self.path_ckpt, 'sane_model_params.ckpt')
        torch.save(pl_module.state_dict(), self.epoch_start_params)
        
        self.epoch_start_optstate = os.path.join(self.path_ckpt, 'sane_optimizer_state.ckpt')
        torch.save(trainer.optimizers[0].state_dict, self.epoch_start_optstate)
    #end
    
    def on_train_epoch_end(self, trainer, pl_module):
        self.last_epoch = pl_module.current_epoch
        
        has_nans = False
        for param in pl_module.parameters():
            if torch.any(param.isnan()):
                has_nans = True
            #end
        #end
        
        if has_nans:
            print('\nNans in model params')
            print('Loading checkpoint model params and optimizer state ...')
            
            sane_model_params = torch.load(self.epoch_start_params)
            pl_module.load_state_dict(sane_model_params)
            
            sane_opt_state = torch.load(self.epoch_start_optstate)
            trainer.optimizers[0].state_dict = sane_opt_state
            
            for param in pl_module.parameters():
                if torch.any(param.isnan()):
                    print('Still nans')
                #end
            #end
        #end
    #end
    
    def get_last_epoch(self):
        return self.last_epoch
    #end
    
    def on_after_backward(self, trainer, pl_module):
        # After loss.backward() and before optimizer.step()
        # Also "on_before_optimizer_step()" is available
        # See pytorch_lightning docs in Callback API
        
        gradients = []
        for param in pl_module.parameters():
            try:
                gradients.append(param.grad.mean())
            except:
                pass
            #end
        #end
        _log_params_grad = torch.mean(torch.Tensor(gradients))
        pl_module.log('param_grad', _log_params_grad, on_step = True, on_epoch = True, prog_bar = False)
    #end
    
    def on_before_zero_grad(self, trainer, pl_module, optimizer):
        
        pass
    #end
    
#end

class Experiment:
    
    def __init__(self, versioning = False, tabula_rasa = False):
        
        # Load configuration file
        load_dotenv(os.path.join(os.getcwd(), 'config.env'))
        with open('./cparams.json', 'r') as f:
            CPARAMS = json.load(f)
        f.close()
        
        # Create a named tuple for the configuration parameters
        _cparams = namedtuple('config_params', CPARAMS)
        cparams  = _cparams(**CPARAMS)
        
        path_data  = os.getenv('PATH_DATA')
        path_model = os.getenv('PATH_MODEL')
        
        self.path_model  = path_model
        self.path_data   = path_data
        self.versioning  = versioning
        self.tabula_rasa = tabula_rasa
        self.cparams     = cparams
        
        self.check_inconstistency()
    #end
    
    def check_inconstistency(self):
        
        if self.cparams.INVERSION == 'fp':
            pass
        elif self.cparams.INVERSION == 'gs':
            pass
        elif self.cparams.INVERSION == 'bl':
            pass
        else:
            raise ValueError('Inversion given not implemented')
        #end
        
        if self.cparams.INVERSION == 'bl' and self.cparams.HR_MASK_SFREQ is not None:
            raise ValueError('Baseline run does not allow HR observations')
        
        if self.cparams.LOAD_CKPT and self.cparams.INVERSION == 'fp':
            raise ValueError('Load ckpt and fixed point, it`s probabily a mistake')
        #end
        
        if not self.cparams.INVERSION == 'fp' and self.cparams.HR_MASK_SFREQ == 1:
            raise ValueError('Reference run requires fixed point')
        #end
        
        if self.versioning and self.cparams.RUNS > 1:
            raise ValueError('Versioning with RUNS > 1 not allowed')
        #end
        
        if self.cparams.HR_MASK_MODE.__class__ is str and self.cparams.HR_MASK_MODE == 'buoy':
            raise ValueError('Specify buoyID if HR_MASK_MODE == "buoy"')
        #end
        
        if self.cparams.HR_MASK_MODE.__class__ is list and self.cparams.HR_MASK_MODE[0] != 'buoy':
            raise ValueError('Specification of in-situ point only if it is a buoy')
        #end
        
        if self.cparams.HR_MASK_MODE == 'buoysMM' and self.cparams.HR_MASK_SFREQ is not None:
            raise ValueError('Multi-modal term with buoys requires not to set sampling frequency for HR. Set it to "null"')
        #end
        
    #end
        
    def initialize_model_names_paths(self, path_manager):
        
        self.path_checkpoint_source, self.name_source_model = path_manager.get_source_ckpt_path()
        self.path_checkpoint = path_manager.get_path('ckpt')
        self.model_name = path_manager.get_model_name()
    #end
    
    def load_checkpoint(self, lit_model, stage, run):
        
        if stage == 'PARAMS-INIT':
            checkpoint_name = os.path.join(self.path_checkpoint_source,
                            f'run{run}-' + self.name_source_model + '-epoch=*.ckpt')
        elif stage == 'TEST':
            checkpoint_name = os.path.join(self.path_checkpoint,
                            f'run{run}-' + self.model_name + '-epoch=*.ckpt')
        #ends
        
        checkpoint_path = glob.glob(checkpoint_name)[0]
        ckpt_model = open(checkpoint_path, 'rb')
        print('\n\nCHECKPOINT ({}) : {}\n\n'.format(stage, checkpoint_path))
        lit_model_statedict = torch.load(ckpt_model)['state_dict']
        lit_model.load_state_dict(lit_model_statedict)
        
        if stage == 'TEST':
            lit_model.eval()
            # lit_model.Phi.eval()
            lit_model.model.eval()
        #end
        
        return lit_model
    #end
    
    def run_simulation(self):
        
        path_manager = PathManager(self.path_model, self.cparams, versioning = self.versioning, tabula_rasa = self.tabula_rasa)
        self.initialize_model_names_paths(path_manager)
        
        self.path_manager = path_manager
        self.path_checkpoint = path_manager.get_path('ckpt')
        
        # Print experiment details
        print('##############################################################')
        print('Experiment\n')
        print('Model name                 : {}'.format(self.model_name))
        print('Prior                      : {}'.format(self.cparams.PRIOR))
        print('Inversion                  : {}'.format(self.cparams.INVERSION))
        print('Trainable varcost params   : {}'.format(self.cparams.LEARN_VC_PARAMS))
        print('Masking mode               : {}'.format(self.cparams.HR_MASK_MODE))
        print('Path source                : {}'.format(self.path_checkpoint_source))
        print('Model source               : {}'.format(self.name_source_model))
        print('Path target                : {}'.format(self.path_checkpoint))
        print('##############################################################')
        
        if self.versioning:
            
            # grid search for optimal hyper-params
            self.main(None)
        else:
            
            # a good choice of hyper-params is set yet
            nruns = 0
            real_run = 0
            while nruns < self.cparams.RUNS:
                
                print('\n***************')
                print(f'RUN       : {nruns+1} / {self.cparams.RUNS}')
                print(f'Effective : {real_run+1} ')
                print(f'Aborted   : {real_run-nruns}')
                self.path_manager.set_nrun(nruns)
                run_outcome = self.main(nruns, real_run)
                
                if run_outcome == 0:
                    self.path_manager.remove_checkpoints(nruns)
                #end
                
                nruns += run_outcome
                real_run += 1
            #end
        #end
    #end
    
    def main(self, run, real_run):
        
        start_time = datetime.datetime.now()
        print('\nRun start at {}\n'.format(start_time))
        
        # DATAMODULE : initialize
        w2d_dm = W2DSimuDataModule(self.path_data, self.cparams)
        
        # MODELS : initialize and configure
        ## Obtain shape data
        shape_data = w2d_dm.get_shapeData()
        land_buoy_coords = w2d_dm.get_land_and_buoy_positions()
        
        ## Instantiate dynamical prior and lit model
        Phi = model_selection(shape_data, self.cparams).to(DEVICE)
        
        if self.cparams.WIND_MODULUS:
            lit_model = LitModel_OSSE1_WindModulus(Phi,
                                                      shape_data,
                                                      land_buoy_coords,
                                                      self.cparams,
                                                      real_run,
                                                      start_time = start_time).to(DEVICE)
        else:
            lit_model = LitModel_OSSE1_WindComponents(Phi, 
                                                      shape_data, 
                                                      land_buoy_coords, 
                                                      self.cparams, 
                                                      real_run, 
                                                      start_time = start_time).to(DEVICE)
        #end
        
        ## Get checkpoint, if needed
        path_ckpt = self.path_manager.get_path('ckpt')
        if self.cparams.LOAD_CKPT:
            lit_model = self.load_checkpoint(lit_model, 'PARAMS-INIT', run)
        #end
        
        # TRAINER : configure properties and callbacks
        profiler_kwargs = {'max_epochs'              : self.cparams.EPOCHS, 
                           'log_every_n_steps'       : 1}
        
        if torch.cuda.is_available():
            profiler_kwargs.update({'accelerator' : 'gpu'})
            profiler_kwargs.update({'devices'     : self.cparams.GPUS})
            profiler_kwargs.update({'precision'   : self.cparams.PRECISION})
        #end
        
        ## Callbacks : model checkpoint and early stopping
        model_checkpoint = ModelCheckpoint(
            monitor    = 'val_loss',
            dirpath    = path_ckpt,
            filename   = f'run{run}-' + self.model_name + '-{epoch:02d}',
            save_top_k = 1,
            mode       = 'min'
        )
        
        early_stopping = EarlyStopping(
            monitor      = 'loss',
            patience     = 50,
            check_finite = True,
        )
        
        # weight_save = SaveWeights(path_ckpt)
        
        # logger = CSVLogger(
        #     save_dir = path_ckpt,
        #     name     = 'csv-log.csv'
        # )
        
        ## Instantiate Trainer
        trainer = pl.Trainer(
            **profiler_kwargs, 
            callbacks = [
                model_checkpoint, 
                early_stopping,
                # weight_save
            ]#,
            # logger = logger
        )
        
        # Train and test
        ## Train
        # trainer.fit(lit_model, datamodule = w2d_dm)
        trainer.fit(lit_model, w2d_dm.train_dataloader(), w2d_dm.val_dataloader())
        
        if lit_model.has_nans():
            
            print('\nNan in model parameters')
            print('Aborting ...\n')
            return_value = 0
            del lit_model
            del Phi
            del trainer
            del w2d_dm
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            #end
            
        else:
            
            ## Test
            lit_model = self.load_checkpoint(lit_model, 'TEST', run)
            lit_model.eval()
            trainer.test(lit_model, datamodule = w2d_dm)
            
            # save reports and reconstructions in the proper target directory
            self.path_manager.save_configfiles(self.cparams, 'config_params')
            self.path_manager.save_model_output(lit_model.get_saved_samples(),
                                                lit_model.mask_land,
                                                self.cparams,
                                                *lit_model.get_learning_curves(),
                                                run)
            lit_model.remove_saved_outputs()
            # self.path_manager.save_litmodel_trainer(lit_model, trainer)
            
            print('\nTraining and test successful')
            print('Returning ...')
            return_value = 1
        #end
        
        end_time = datetime.datetime.now()
        print('\nRun end at {}\n'.format(end_time))
        print('Run time = {}\n'.format(end_time - start_time))
        
        return return_value
    #end
#end


if __name__ == '__main__':
    
    # qui argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-vr', type = int, default = 0)
    parser.add_argument('-tr', type = int, default = 0)
    
    args = parser.parse_args()
    versioning  = args.vr
    tabula_rasa = args.tr
    
    if versioning > 1:  versioning = 1
    if tabula_rasa > 1: tabula_rasa = 1
    
    versioning  = bool(versioning)
    tabula_rasa = bool(tabula_rasa)
    
    exp = Experiment(versioning = versioning, tabula_rasa = tabula_rasa)
    exp.run_simulation()
#end

