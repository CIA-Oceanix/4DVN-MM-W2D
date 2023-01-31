import os
import glob
import datetime
import json
import torch
import pickle
import netCDF4 as nc
import numpy as np



def get_model_name(cparams, versioning = False):
    
    # MODEL  NAME
    # format : 4DVN-W2D-<mask_HR>[-ckpt-gs_n_itref]-<inversion>-<lr_hr_sfreqs / "REFRUN">-<prior>
    model_name = f'{cparams.VNAME}'
    
    hr_mask_mode = cparams.HR_MASK_MODE
    if hr_mask_mode.__class__ is list:
        mode = hr_mask_mode[0]
        nbuoy = hr_mask_mode[1]
        model_name = f'{model_name}-{mode}{nbuoy}'
    else:
        model_name = f'{model_name}-{cparams.HR_MASK_MODE}'
    #end
    
    if cparams.MM_OBSMODEL:
        model_name += '-MM'
    #end
    
    if cparams.HR_MASK_SFREQ == 1:
        
        exp_osse_1_specs = 'REFRUN'
    else:
        
        # parse LR description
        lr_sampling_freq = cparams.LR_MASK_SFREQ
        if lr_sampling_freq is None:
            lr_sampling_freq_tag = '0'
        else:
            if lr_sampling_freq.__class__ is list:
                
                lr_sampling_freq_tag = '_'
                for item in lr_sampling_freq:
                    lr_sampling_freq_tag += (str(item) + '_')
                #end
            
            else:
                lr_sampling_freq_tag = '{}'.format(lr_sampling_freq)
            #end
        #end
        
        if cparams.LR_SAMP_DELAY:
            lr_sampling_freq_tag += 'rd'
        #end
        
        if cparams.LR_INTENSITY:
            lr_sampling_freq_tag += 'ri'
        #end
        
        # parse HR description
        hr_sampling_freq = cparams.HR_MASK_SFREQ
        if hr_sampling_freq is None:
            hr_sampling_freq_tag = '0'
        else:
            if hr_sampling_freq.__class__ is list:
                
                hr_sampling_freq_tag = '_'
                for item in hr_sampling_freq:
                    hr_sampling_freq_tag += (str(item) + '_')
                #end
                
            else:
                hr_sampling_freq_tag = '{}'.format(hr_sampling_freq)
            #end
        #end
        
        exp_osse_1_specs = f'sflr{lr_sampling_freq_tag}-sfhr{hr_sampling_freq_tag}'
    #end
    
    if cparams.INVERSION == 'gs':
        
        nsol_iter = cparams.NSOL_ITER
        nsol_iter_ref = cparams.NSOL_IT_REF
        
        if cparams.LOAD_CKPT:
            model_source = f'{model_name}-gs{nsol_iter_ref}it-{exp_osse_1_specs}-{cparams.PRIOR}'
            inversion = f'ckpt-gs{nsol_iter_ref}it-gs{nsol_iter}it'
        else:
            model_source = None
            inversion = f'gs{nsol_iter}it'
        #end
        
    elif cparams.INVERSION == 'fp':
        model_source = None
        inversion = 'fp1it'
    
    elif cparams.INVERSION == 'bl':
        model_source = None
        inversion = 'baseline'
    #end
    
    if cparams.INVERSION == 'bl':
        model_name = f'{cparams.VNAME}-baseline-sflr{lr_sampling_freq_tag}-{cparams.PRIOR}'
    else:
        model_name += f'-{inversion}-{exp_osse_1_specs}-{cparams.PRIOR}'
    #end
    return model_name, model_source
#end

class PathManager:
    
    def __init__(self, mother_dir, cparams, versioning = False, tabula_rasa = False):
        
        if tabula_rasa:
            os.system(r'rm -rf {}/*'.format(mother_dir))
        #end
        
        #---------------------------------------------------------------------------------------------
        model_name, model_source = get_model_name(cparams, versioning)
        
        if model_source is not None:
            self.path_ckpt_source = os.path.join(mother_dir, model_source, 'ckpt')
        else:
            self.path_ckpt_source = None
        #end
        self.model_source = model_source
        
        if versioning:
            time_now = datetime.datetime.now()
            unique_token = '-' + time_now.strftime('%Y%m%d%H%M%S')
            version_path = os.path.join(mother_dir, model_name + unique_token)
        else:
            version_path = os.path.join(mother_dir, model_name)
        #end
        # END  MODEL NAME
        #---------------------------------------------------------------------------------------------
        
        # create version home directory
        if not os.path.exists(version_path):
            os.mkdir(version_path)
        #end
        
        # directories hierarchy
        path_ckpt = os.path.join(version_path, 'ckpt')
        if not os.path.exists(path_ckpt):
            os.mkdir(path_ckpt)
        #end
        
        path_litmodel_trainer = os.path.join(version_path, 'litmodel_trainer')
        if not os.path.exists(path_litmodel_trainer):
            os.mkdir(path_litmodel_trainer)
        #end
        
        path_evalmetrics = os.path.join(version_path, 'evalmetrics')
        if not os.path.exists(path_evalmetrics):
            os.mkdir(path_evalmetrics)
        #end
        
        path_configfiles = os.path.join(version_path, 'configfiles')
        if not os.path.exists(path_configfiles):
            os.mkdir(path_configfiles)
        #end
        
        path_modeloutput = os.path.join(version_path, 'modeloutput')
        if not os.path.exists(path_modeloutput):
            os.mkdir(path_modeloutput)
        #end
        
        self.model_name            = model_name
        self.path_ckpt             = path_ckpt
        self.path_litmodel_trainer = path_litmodel_trainer
        self.path_evalmetrics      = path_evalmetrics
        self.path_configfiles      = path_configfiles
        self.path_modeloutput      = path_modeloutput
        self.nrun = None
        
        self.remove_checkpoints()
        self.initialize_netCDF4_dataset(cparams.REGION_EXTENT_PX, cparams.RUNS)
    #end
    
    def get_model_name(self):
        return self.model_name
    #end
    
    def get_source_ckpt_path(self):
        return self.path_ckpt_source, self.model_source
    #end
    
    def remove_checkpoints(self, run_to_remove = None):
        
        if run_to_remove is None:
            try:
                os.system(r'rm -rf {}/csv-log.csv'.format(self.path_ckpt))
            except:
                pass
            #end
            os.system(r'rm -f {}/*'.format(self.path_ckpt))
            print('REMOVE CKPTS in {}'.format(self.path_ckpt))
        else:
            ckpt_to_remove = glob.glob(os.path.join(self.path_ckpt, f'run{run_to_remove}*'))[0]
            os.remove(ckpt_to_remove)
        #end
    #end
    
    def initialize_netCDF4_dataset(self, region_extent, runs):
        
        if os.path.exists(os.path.join(self.path_modeloutput, 'reconstructions.nc')):
            os.remove(os.path.join(self.path_modeloutput, 'reconstructions.nc'))
            print('Old dataset.nc removed ...')
        #end
        
        dataset = nc.Dataset(os.path.join(self.path_modeloutput, 'reconstructions.nc'), 'w', format = 'NETCDF4_CLASSIC')
        dataset.createDimension('extent', region_extent)
        dataset.createDimension('time', 24)
        dataset.createDimension('one', 1)
        dataset.createDimension('run', runs)
        dataset.createDimension('batch', None)
        
        dataset.createVariable('reco', np.float32, ('run', 'batch', 'time', 'extent', 'extent'))
        dataset.createVariable('data', np.float32, ('one', 'batch', 'time', 'extent', 'extent'))
        dataset.createVariable('mask', np.float32, ('one', 'extent', 'extent'))
        dataset.close()
        print('Dataset.nc initialized and closed ...')
    #end
    
    def get_path(self, directory, absolute = False):
        
        if directory == 'ckpt':
            path = self.path_ckpt
        elif directory == 'litmodel_trainer':
            path = self.path_litmodel_trainer
        elif directory == 'evalmetrics':
            path = self.path_evalmetrics
        elif directory == 'configfiles':
            path = self.path_configfiles
        elif directory == 'modeloutput':
            path = self.path_modeloutput
        else:
            raise ValueError('No match for directory in local filesystem')
        #end
        
        if absolute:
            return os.path.abspath(path)
        else:
            return path
        #end
    #end
    
    def set_nrun(self, nrun):
        
        self.nrun = nrun
    #end
    
    def print_evalreport(self, report_dict):
        
        fname = 'evalmetrics' if self.nrun is None else f'evalmetrics_{self.nrun}'
        
        with open(os.path.join(self.path_evalmetrics, f'{fname}.json'), 'w') as f:
            json.dump(report_dict, f, indent = 4)
        f.close()
    #end
    
    def save_configfiles(self, configfile, configfile_name):
        
        with open(os.path.join(self.path_configfiles, 
                               configfile_name + '.json'), 'w') as f:
            json.dump(configfile._asdict(), f, indent = 4)
        f.close()
        
        with open(os.path.join(self.path_configfiles,
                               'modelname.txt'), 'w') as f:
            f.write(self.model_name)
        f.close()
    #end
    
    def save_litmodel_trainer(self, lit_model, trainer):
        
        # with open(os.path.join(self.path_litmodel_trainer, 'lit_model.pkl'), 'wb') as f:
        #     torch.save(lit_model.cpu(), f)
        # f.close()
        
        # with open(os.path.join(self.path_litmodel_trainer, 'trainer.pkl'), 'wb') as f:
        #     pickle.dump(trainer, f)
        # #end
        
        path_models_statedicts = os.path.join(self.path_litmodel_trainer, 'models_statedicts')
        if not os.path.exists(path_models_statedicts):
            os.mkdir(path_models_statedicts)
        #end
        
        with open(os.path.join(path_models_statedicts, 'Phi.pkl'), 'wb') as f:
            torch.save(lit_model.Phi.state_dict(), f)
        f.close()
        
        with open(os.path.join(path_models_statedicts, 'ObsModel.pkl'), 'wb') as f:
            torch.save(lit_model.observation_model.state_dict(), f)
        f.close()
        
        with open(os.path.join(path_models_statedicts, 'GradSolverLSTM.pkl'), 'wb') as f:
            torch.save(lit_model.model.model_Grad.state_dict(), f)
        f.close()
        
        with open(os.path.join(path_models_statedicts, 'VarCost.pkl'), 'wb') as f:
            torch.save(lit_model.model.model_VarCost.state_dict(), f)
        f.close()
    #end
    
    def save_model_output(self, outputs, mask_land, cparams, train_losses, val_losses, run):
        
        data = torch.cat([item['data'] for item in outputs], dim = 0)
        reco = torch.cat([item['reco'] for item in outputs], dim = 0)
        
        reco_ncd = nc.Dataset(os.path.join(self.path_modeloutput, 'reconstructions.nc'), 'a')
        if run == 0:
            reco_ncd['data'][0,:,:,:,:] = data
            reco_ncd['mask'][0,:,:] = mask_land.cpu()
        #end
        reco_ncd['reco'][run,:,:,:,:] = reco
        reco_ncd.close()
        
        with open(os.path.join(self.path_modeloutput,'cparams.json'), 'w') as f:
            json.dump(cparams._asdict(), f, indent = 4)
        f.close()
        
        with open(os.path.join(self.path_modeloutput, 'learning_curves.pkl'), 'wb') as f:
            pickle.dump({'train' : train_losses, 'val' : val_losses}, f)
        f.close()
    #end
#end