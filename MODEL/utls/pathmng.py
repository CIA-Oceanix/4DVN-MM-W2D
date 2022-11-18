import os
import datetime
import json
import torch
import pickle
import netCDF4 as nc
import numpy as np



def get_model_name(cparams, versioning):
    
    pass
#end

class PathManager:
    
    def __init__(self, mother_dir, cparams, versioning = True, tabula_rasa = False):
        
        if tabula_rasa:
            os.system(r'rm -rf {}/*'.format(mother_dir))
        #end
        
        #---------------------------------------------------------------------------------------------
        # MODEL  NAME
        # format : 4DVN-W2D-<mask_HR>[-ckpt-gs_n_itref]-<inversion>-<lr_hr_sfreqs / "REFRUN">-<prior>
        hr_mask_mode = cparams.HR_MASK_MODE
        if hr_mask_mode.__class__ is list:
            mode = hr_mask_mode[0]
            nbuoy = hr_mask_mode[1]
            model_name = f'{cparams.VNAME}-{mode}{nbuoy}'
        else:
            model_name = f'{cparams.VNAME}-{cparams.HR_MASK_MODE}'
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
        
        if cparams.GS_TRAIN and not versioning:
            
            nsol_iter = cparams.NSOL_ITER
            nsol_iter_ref = cparams.NSOL_IT_REF
            
            if cparams.LOAD_CKPT:
                model_source = f'{model_name}-gs{nsol_iter_ref}it-{exp_osse_1_specs}-{cparams.PRIOR}'
                path_ckpt_source = os.path.join(mother_dir, model_source, 'ckpt')
                inversion = f'ckpt-gs{nsol_iter_ref}it-gs{nsol_iter}it'
            else:
                model_source = None
                path_ckpt_source = None
                inversion = f'gs{nsol_iter}it'
            #end
            
        else:
            model_source = None
            path_ckpt_source = None
            inversion = 'fp1it'
        #end
        
        self.path_ckpt_source = path_ckpt_source
        self.model_source = model_source
        model_name += f'-{inversion}-{exp_osse_1_specs}-{cparams.PRIOR}'
        
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
        
        # For security, remove all the saved checkpoints in order no to load the wrong one
        os.system(r'rm -f {}/*'.format(path_ckpt))
        print('REMOVE CKPTS in {}'.format(path_ckpt))
        
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
        
        self.initialize_netCDF4_dataset(cparams.REGION_EXTENT_PX, cparams.RUNS)
    #end
    
    def get_model_name(self):
        return self.model_name
    #end
    
    def get_source_ckpt_path(self):
        return self.path_ckpt_source, self.model_source
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
        
        with open(os.path.join(self.path_litmodel_trainer, 'lit_model.pkl'), 'wb') as f:
            torch.save(lit_model.cpu(), f)
        f.close()
        
        with open(os.path.join(self.path_litmodel_trainer, 'trainer.pkl'), 'wb') as f:
            pickle.dump(trainer, f)
        #end
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