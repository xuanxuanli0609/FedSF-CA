import os
import torch
import sys
from collections import OrderedDict
from . import networks

class BaseModel(torch.nn.Module):
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if self.isTrain:
            self.model_names = []
            self.visual_names=[]
            self.loss_names=[]
            # initialize optimizers
            self.optimizers = []

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    def get_file_path(self):
        return self.images_path 
    # used in test time, no backprop
    def test(self,label, inst, image, feat, c_idx):
        with torch.no_grad():
            self.inference(label, inst, image, feat)
            #self.inference(label, inst, image, feat, c_idx)
    # return visualization images. train.py will display these images, and save the images to a html
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name).detach()
        return visual_ret

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass


    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s.pkl' % (epoch_label)
        if not save_dir:
            save_dir = self.save_dir
       
        save_path = os.path.join(save_dir, save_filename)   
        print(save_path)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:   
                pretrained_dict = torch.load(save_path)                
                model_dict = network.state_dict()
                try:
                    name = 'net'+network_label
                    pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in pretrained_dict.items() if k.split('.')[0]==name}                    
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)                  

    def update_learning_rate():
        pass

    def setup(self, opt):
        if self.isTrain:
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = 'iter_%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks(opt.verbose)
