
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import EarlyStopping

from pathmng import PathManager
from dlmodels import LitModel, model_selection
from dutls import W2DSimuDataModule

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # print('Try to free CUDA cache')
    # torch.cuda.empty_cache()
    gpus = -1
    print('Program runs using device : {}\n'.format(DEVICE))
else:
    DEVICE = torch.device('cpu')
    gpus = 0
    print('Program runs using device : {}'.format(DEVICE))
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
        
        self.check_inconstistency()
    #end
    
    def check_inconstistency(self):
        
        if self.cparams.GS_TRAIN and self.cparams.FIXED_POINT:
            raise ValueError('Either fixed point or gradient train')
        #end
        
        if self.cparams.LOAD_CKPT and self.cparams.FIXED_POINT:
            raise ValueError('Load ckpt and fixed point, it`s probabily a mistake')
        #end
    
    def get_model(self):
        
        model_name = f'{self.cparams.VNAME}-{self.cparams.HR_MASK_MODE}'
        
        if self.cparams.GS_TRAIN and not self.versioning:
            
            n_iter_ref = self.cparams.NSOL_IT_REF
            n_iter     = self.cparams.NSOL_ITER
            
            if self.cparams.LOAD_CKPT:
                mname_source = model_name + f'-gs{n_iter_ref}it-{self.cparams.PRIOR}'
                mname_target = model_name + f'-ckpt-gs{n_iter_ref}it-gs{n_iter}it'
                
                # instantiate a path_manager only to get the
                # path to checkpoints of source version
                path_manager_source = PathManager(self.path_model,
                                                  mname_source,
                                                  self.cparams.LOAD_CKPT,
                                                  versioning  = False,
                                                  tabula_rasa = False)
                self.path_checkpoint_source = path_manager_source.get_path('ckpt')
                self.name_source_model = mname_source
                
                model_name = mname_target
            else:
                
                self.path_checkpoint_source = 'None'
                self.name_source_model = 'None'
                model_name += f'-gs{n_iter}it'
            #end
        else:
            self.path_checkpoint_source = None
            self.name_source_model = None
            model_name += '-fp1it'
        #end
        
        model_name += f'-{self.cparams.PRIOR}'
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
                                   self.cparams.LOAD_CKPT,
                                   versioning  = self.versioning,
                                   tabula_rasa = self.tabula_rasa)
        self.path_manager = path_manager
        self.path_checkpoint = path_manager.get_path('ckpt')
        
        # Print experiment details
        print('##############################################################')
        print('Experiment\n')
        print('Model name                 : {}'.format(self.model_name))
        print('Prior                      : {}'.format(self.cparams.PRIOR))
        print('Fixed point                : {}'.format(self.cparams.FIXED_POINT))
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
            for _run in range(self.cparams.RUNS):
                
                print(f'Run : {_run}')
                self.path_manager.set_nrun(_run)
                self.main(_run)
            #end
        #end
    #end
    
    def main(self, run):
        
        start_time = datetime.datetime.now()
        print('\nRun start at {}\n'.format(start_time))
        
        # DATAMODULE : initialize
        w2d_dm = W2DSimuDataModule(self.path_data, self.cparams)
        
        # MODELS : initialize and configure
        ## Obtain shape data
        shape_data = w2d_dm.get_shapeData()
        
        ## Instantiate dynamical prior and lit model
        Phi = model_selection(shape_data, self.cparams).to(DEVICE)
        lit_model = LitModel(Phi, shape_data, self.cparams).to(DEVICE)
        
        ## Get checkpoint, if needed
        path_ckpt = self.path_manager.get_path('ckpt')
        if self.cparams.LOAD_CKPT:
            
            lit_model = self.load_checkpoint(lit_model, 'PARAMS-INIT', run)
            name_append = '_second'
        else:
            name_append = ''
            pass
        #end
        
        # TRAINER : configure properties and callbacks
        profiler_kwargs = {'max_epochs'        : self.cparams.EPOCHS, 
                           'log_every_n_steps' : 1}
        
        if torch.cuda.is_available():
            profiler_kwargs.update({'gpus'        : gpus})
            # profiler_kwargs.update({'accelerator' : 'ddp'})
            profiler_kwargs.update({'precision'   : self.cparams.PRECISION})
        #end
        
        ## Callbacks : model checkpoint and early stopping
        model_checkpoint = ModelCheckpoint(
            monitor    = 'val_loss',
            dirpath    = path_ckpt,
            filename   = f'run{run}-' + self.model_name + '-{epoch:02d}' + name_append,
            save_top_k = 1,
            mode       = 'min'
        )
        
        early_stopping = EarlyStopping(
            monitor  = 'val_loss',
            mode     = 'min',
            patience = 50
        )
        
        ## Instantiate Trainer
        trainer = pl.Trainer(**profiler_kwargs, callbacks = [model_checkpoint, early_stopping])
        
        # Train and test
        ## Train
        # trainer.fit(lit_model, w2d_dm.train_dataloader(), w2d_dm.val_dataloader())
        trainer.fit(lit_model, w2d_dm)
        
        ## Test
        lit_model = self.load_checkpoint(lit_model, 'TEST', run)
        lit_model.eval()
        # trainer.test(lit_model, w2d_dm.test_dataloader())
        trainer.test(lit_model, w2d_dm)
        test_loss = lit_model.get_test_loss()
        print('\n\nTest loss = {}\n\n'.format(test_loss))
        
        # perf_dict_metrics = lit_model.get_eval_metrics()
        # perf_dict_metrics.update({'mse_test' : test_loss.item()})
        
        print('Mean data lr = {}'.format(torch.Tensor(lit_model.means_data_lr).mean()))
        print('Mean data an = {}'.format(torch.Tensor(lit_model.means_data_an).mean()))
        print('Mean reco lr = {}'.format(torch.Tensor(lit_model.means_reco_lr).mean()))
        print('Mean reco an = {}'.format(torch.Tensor(lit_model.means_reco_an).mean()))
        
        # save reports and reconstructions in the proper target directory
        self.path_manager.save_configfiles(self.cparams, 'config_params')
        self.path_manager.save_model_output(lit_model.get_saved_samples(),
                                            self.cparams,
                                            *lit_model.get_learning_curves())
        lit_model.remove_saved_outputs()
        self.path_manager.save_litmodel_trainer(lit_model, trainer)
        # self.path_manager.print_evalreport(perf_dict_metrics)
        
        end_time = datetime.datetime.now()
        print('\nRun end at {}\n'.format(end_time))
        print('Run time = {}\n'.format(end_time - start_time))
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

