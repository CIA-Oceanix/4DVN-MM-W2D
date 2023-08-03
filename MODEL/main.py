
import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'utls'))
sys.path.append(os.path.join(os.getcwd(), '4dvar'))

from dotenv import load_dotenv
import datetime
import glob
import gc
import json
from collections import namedtuple
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from pathmng import PathManager
from litmodels import LitModel_OSSE1_WindModulus, LitModel_OSSE2_Distribution
from dlmodels import model_selection
from dutls import W2DSimuDataModule, WPDFSimuDataModule

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print('Program runs using device : {}\n'.format(DEVICE))
else:
    DEVICE = torch.device('cpu')
    print('Program runs using device : {}\n'.format(DEVICE))
#end
torch.autograd.set_detect_anomaly(True)
# torch.manual_seed(161020)


class Experiment:
    
    def __init__(self):
        
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
        
        if self.cparams.TABULA_RASA and not self.cparams.VERSIONING:
            raise ValueError('Attention! TABULA_RASA set True and VERSIONING False. This will delete all results')
        #end
        
        if self.cparams.INVERSION == 'bl' and self.cparams.HR_MASK_SFREQ is not None:
            raise ValueError('Baseline run does not allow HR observations')
        
        if self.cparams.LOAD_CKPT and self.cparams.INVERSION == 'fp':
            raise ValueError('Load ckpt and fixed point, it`s probabily a mistake')
        #end
        
        if not self.cparams.INVERSION == 'fp' and self.cparams.HR_MASK_SFREQ == 1:
            raise ValueError('Reference run requires fixed point')
        #end
        
        if self.cparams.VERSIONING and self.cparams.RUNS > 1:
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
        
        if (self.cparams.INVERSION == 'fp' or self.cparams.INVERSION == 'bl') and self.cparams.MM_OBSMODEL:
            raise ValueError('Trainable observation operator is to be used only with "gs" inversion')
        #end
        
        # if self.cparams.VNAME == '4DVN-PDF' and self.cparams.PRIOR != 'FPN':
        #     raise ValueError('Select FPN prior with 4DVN-PDF case')
        # #end
        
    #end
    
    def print_exp_details(self):
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
            lit_model.model.eval()
        #end
        
        return lit_model
    #end
    
    def run_simulation(self):
        
        path_manager = PathManager(self.path_model, self.cparams)
        self.initialize_model_names_paths(path_manager)
        
        self.path_manager = path_manager
        self.path_checkpoint = path_manager.get_path('ckpt')    
        
        # introduce
        self.print_exp_details()
        
        # DATAMODULE : initialize
        if self.cparams.VNAME == '4DVN-W2D':
            self.w2d_dm = W2DSimuDataModule(self.path_data, self.cparams)
        elif self.cparams.VNAME == '4DVN-PDF':
            self.w2d_dm = WPDFSimuDataModule(self.path_data, self.cparams)
        #end
        
        if self.cparams.VERSIONING:
            
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
        
        # MODELS : initialize and configure
        ## Obtain shape data
        shape_data = self.w2d_dm.get_shapeData()
        land_buoy_coords = self.w2d_dm.get_land_and_buoy_positions()
        
        ## Instantiate dynamical prior and lit model
        if self.cparams.VNAME == '4DVN-W2D':
            if self.cparams.WIND_MODULUS:
                
                Phi = model_selection(shape_data, self.cparams).to(DEVICE)
                lit_model = LitModel_OSSE1_WindModulus(Phi,
                                                       shape_data,
                                                       land_buoy_coords,
                                                       self.cparams,
                                                       real_run,
                                                       start_time = start_time).to(DEVICE)
            else:
                # COMPONENTS DEPRECATED
                pass
            #end
        elif self.cparams.VNAME == '4DVN-PDF':
            
            normparams = self.w2d_dm.get_normparams(stage = 'train')
            Phi = model_selection(shape_data, self.cparams, normparams).to(DEVICE)
            lit_model = LitModel_OSSE2_Distribution(Phi,
                                                    shape_data,
                                                    land_buoy_coords,
                                                    normparams,
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
        
        ## Instantiate Trainer
        trainer = pl.Trainer(
            **profiler_kwargs, 
            callbacks = [
                model_checkpoint, 
                early_stopping,
            ]
        )
        
        # Train and test
        ## Train
        trainer.fit(lit_model, datamodule = self.w2d_dm)
        
        if lit_model.has_nans():
            
            print('\nNan in model parameters')
            print('Aborting ...\n')
            return_value = 0
            del lit_model
            del Phi
            del trainer
            gc.collect()
            
        else:
            
            ## Test
            lit_model = self.load_checkpoint(lit_model, 'TEST', run)
            lit_model.eval()
            trainer.test(lit_model, datamodule = self.w2d_dm)
            
            # save reports and reconstructions in the proper target directory
            self.path_manager.save_configfiles(self.cparams, 'config_params')
            self.path_manager.save_model_output(lit_model.get_saved_samples(),
                                                lit_model.mask_land,
                                                self.cparams,
                                                *lit_model.get_learning_curves(),
                                                run)
            lit_model.remove_saved_outputs()
            
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
    
    exp = Experiment()
    exp.run_simulation()
#end


