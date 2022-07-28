
import os
import datetime


class PathManager:
    
    def __init__(self, mother_dir, model_name, versioning = True):
        
        if versioning:
            time_now = datetime.datetime.now()
            unique_token = '-' + time_now.strftime('%Y%m%d%H%M%S')
            version_path = os.path.join(mother_dir, model_name + unique_token)
        else:
            version_path = os.path.join(mother_dir, model_name)
        #end
        
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
        
        self.path_ckpt             = path_ckpt
        self.path_litmodel_trainer = path_litmodel_trainer
        self.path_evalmetrics      = path_evalmetrics
        self.path_configfiles      = path_configfiles
        self.path_modeloutput      = path_modeloutput
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
#end


class VersionFinder:
    
    def __init__(self):
        pass
    #end
    
#end
    
        
        