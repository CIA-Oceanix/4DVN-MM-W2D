
print('###################################')
print('Experiment : ')
print('###################################')

import os
import sys
sys.path.append(os.path.join(os.getcwd(), 'utls'))
sys.path.append(os.path.join(os.getcwd(), '4dvar'))

from dotenv import load_dotenv
import glob
import json
import argparse
from collections import namedtuple
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from pathmng import PathManager
from dlmodels import LitModel, Phi_r
from dutls import W2DSimuDataModule

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    gpus = -1
else:
    DEVICE = torch.device('cpu')
    gpus = 0
#end

class Experiment:
    
    def __init__(self, versioning = True):
        
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
    #end
    
    def get_model(self):
        
        model_name = self.cparams.VNAME
        
        if self.cparams.GS_TRAIN and not self.versioning:
            
            n_iter_ref = self.cparams.NSOL_IT_REF
            n_iter     = self.cparams.NSOL_ITER
            
            if self.cparams.LOAD_CKPT:
                mname_source = model_name + f'-gs{n_iter_ref}it'
                mname_target = model_name + f'-ckpt-gs{n_iter_ref}it-gs{n_iter}it'
                
                # instantiate a path_manager only to get the
                # path to checkpoints of source version
                path_manager_source = PathManager(self.path_model,
                                                  mname_source, 
                                                  versioning  = False,
                                                  tabula_rasa = False)
                self.path_checkpoint_source = path_manager_source.get_path('ckpt')
                self.name_source_model = mname_source
                
                model_name = mname_target
            else:
                
                model_name += f'-gs{n_iter}it'
            #end
        #end
        
        self.model_name = model_name
        return model_name
    #end
    
    def load_checkpoint(self, lit_model, stage, run):
        
        if self.cparams.LOAD_CKPT:
            checkpoint_name = os.path.join(self.path_checkpoint_source,
                            f'run{run}-' + self.name_source_model + '-epoch=*.ckpt')
        else:
            checkpoint_name = os.path.join(self.path_checkpoint,
                            f'run{run}-' + self.model_name + '-epoch=*.ckpt')
        #end
        
        checkpoint_path = glob.glob(checkpoint_name)[0]
        ckpt_model = open(checkpoint_path, 'rb')
        print('\n\nCHECKPOINT ({}) : {}\n\n'.format(stage, checkpoint_path))
        lit_model_statedict = torch.load(ckpt_model)['state_dict']
        lit_model.load_state_dict(lit_model_statedict)
        
        if stage == 'TEST':
            lit_model.eval()
            lit_model.Phi.eval()
            lit_model.model.eval()
        #end
        
        return lit_model
    #end
    
    def run_simulation(self):
        
        if self.versioning and self.cparams.RUNS > 1:
            raise ValueError('Versioning with RUNS > 1 not allowed')
        #end
        
        model_name = self.get_model()
        
        # instantiate path_manager of this version
        path_manager = PathManager(self.path_model,
                                   model_name,
                                   versioning  = self.versioning,
                                   tabula_rasa = self.tabula_rasa)
        self.path_manager = path_manager
        self.path_checkpoint = path_manager.get_path('ckpt')
        
        if self.versioning:
            
            # grid search for optimal hyper-params
            self.main(None)
        else:
            
            # a good choice of hyper-params is set yet
            for _run in range(self.cparams.RUNS):
                
                print(f'Run : {_run}')
                self.path_manager.set_nrun(_run)
                self.main(_run)
            #end
        #end
    #end
    
    def main(self, run):
        
        # DATAMODULE : initialize
        w2d_dm = W2DSimuDataModule(self.path_data, self.cparams.BATCH_SIZE)
        print(w2d_dm.__class__)
        
        # MODELS : initialize and configure
        ## Obtain shape data
        # batch_size, ts_length, height, width
        shape_data = tuple( next(iter(w2d_dm.train_dataloader())).shape )
        
        
        ## Instantiate dynamical prior and lit model
        Phi = Phi_r(shape_data, self.cparams)
        lit_model = LitModel(Phi, shape_data, self.cparams)
        print(lit_model.__class__)
        
        ## Get checkpoint, if needed
        path_ckpt = self.path_manager.get_path('ckpt')
        if self.cparams.LOAD_CKPT:
            
            lit_model = self.load_checkpoint(lit_model, 'LOAD', run)
            name_append = '_second'
        else:
            name_append = ''
            pass
        #end
        
        # TRAINER : configure properties and callbacks
        profiler_kwargs = {'max_epochs'        : self.cparams.EPOCHS, 
                           'log_every_n_steps' : 1,
                           'gpus'              : gpus}
        
        ## Checkpoint callback
        model_checkpoint = ModelCheckpoint(
            monitor    = 'val_loss',
            dirpath    = path_ckpt,
            filename   = f'run{run}-' + self.model_name + '-{epoch:02d}' + name_append,
            save_top_k = 1,
            mode       = 'min'
        )
        
        ## Instantiate Trainer
        trainer = pl.Trainer(**profiler_kwargs, callbacks = [model_checkpoint])
        
        # Train and test
        ## Train
        trainer.fit(lit_model, w2d_dm.train_dataloader(), w2d_dm.val_dataloader())
        
        ## Test
        lit_model = self.load_checkpoint(lit_model, 'TEST', run)
        # trainer.test(lit_model, w2d_dm.test_dataloader())
        
        # print report in the proper target directory
        self.path_manager.save_configfiles(self.cparams, 'config_params')
        
        perf_dict = {
            'mse' : np.random.normal(0, 1),
            'kld' : np.random.normal(0, 1)
        }
        self.path_manager.print_evalreport(perf_dict)
        
        self.path_manager.save_litmodel_trainer(lit_model, trainer)
        
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
    
    exp = Experiment(versioning = versioning)
    exp.run_simulation()
#end
