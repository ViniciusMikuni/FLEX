#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:11:28 2024

@author: ben
"""

import numpy as np
import torch
import h5py
import os
import torch.nn.functional as F
import random
from torchvision import datasets, transforms
from abc import ABC, abstractmethod

class AddNoise:
    def __init__(self, p=0.1, add_noise_level=0.2, mult_noise_level=0.1):
        self.p = p
        self.add_noise_level = add_noise_level
        self.mult_noise_level = mult_noise_level

    def _noise(self, x):
        add_noise = 0.0
        mult_noise = 1.0
        if self.add_noise_level > 0.0:
            add_noise = self.add_noise_level * np.random.beta(2, 5) * torch.randn_like(x).to(x.device)
        if self.mult_noise_level > 0.0:
            mult_noise = self.mult_noise_level * np.random.beta(2, 5) * (2 * torch.rand_like(x).to(x.device) - 1) + 1
        return mult_noise * x + add_noise

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            x = self._noise(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, add_noise_level={self.add_noise_level}, mult_noise_level={self.mult_noise_level})"



class FlipAugmentation:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, im1, im2, im3, direction=None):
        """
        Perform random flipping on three input tensors.

        Args:
            im1, im2, im3 (torch.Tensor): Tensors of shape (1, H, W) to be flipped.
            direction (str, optional): Direction of the flip ('horizontal' or 'vertical'). If None, it is chosen randomly.

        Returns:
            tuple: Three tensors of shape (1, H, W), possibly flipped.
        """
        assert im1.shape[0] == 1 and im2.shape[0] == 1 and im3.shape[0] == 1, \
            "All inputs must have shape (1, H, W)."
        assert im1.ndim == 3 and im2.ndim == 3 and im3.ndim == 3, \
            "All inputs must be 3-dimensional tensors."

        if random.random() < self.p:
            if direction == 'horizontal':
                im1 = torch.flip(im1, dims=[2])  # Flip along width (W)
                im2 = torch.flip(im2, dims=[2])
                im3 = torch.flip(im3, dims=[2])
            elif direction == 'vertical':
                im1 = torch.flip(im1, dims=[1])  # Flip along height (H)
                im2 = torch.flip(im2, dims=[1])
                im3 = torch.flip(im3, dims=[1])
            else:
                raise ValueError("Direction must be 'horizontal', 'vertical', or None.")

        return im1, im2, im3    



class DataLoader(torch.utils.data.Dataset, ABC):
    def __init__(self,
                 factor = 8,
                 num_pred_steps=1,
                 patch_size=256,
                 stride = 64,
                 train=True,
                 scratch_dir='./',
                 oversampling = 40,
                 cond_snapshots = 2,
                 ):


        self.factor = factor
        self.num_pred_steps = num_pred_steps
        self.train = train
        self.oversampling = oversampling
        self.patch_size = patch_size
        self.stride = stride
        self.cond_snapshots = cond_snapshots
        self.datasets = self.open_hdf5()

        self.data_shape = self.datasets[0].shape
        self.max_row = (self.data_shape[-2] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[-1] - self.patch_size) // self.stride + 1
        

        self.mean, self.std = self.get_norm()
        if self.train:
            self.transform = AddNoise(0.1)
        else:
            self.transform = AddNoise(0.0)
            

    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()
                        
        # Select a time index 
        index = index // self.oversampling + self.cond_snapshots - 1
        
        # Randomly select a patch from the image
        patch_row = np.random.randint(0, self.max_row) * self.stride
        patch_col = np.random.randint(0, self.max_col) * self.stride
        
        #Select one of the training files
        random_dataset = np.random.randint(0, len(self.paths))
        
        Reynolds_number = self.RN[random_dataset]
        dataset = self.datasets[random_dataset]

        if len(self.data_shape) == 4: #Climate dataset has one additional dimension for task
            patch = torch.from_numpy(dataset[index - self.cond_snapshots + 1:index + 1,2, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float()
            future_patch = torch.from_numpy(dataset[index + self.num_pred_steps,2, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
        else:
            patch = torch.from_numpy(dataset[index - self.cond_snapshots + 1:index + 1, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float()
            future_patch = torch.from_numpy(dataset[index + self.num_pred_steps, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0)
            

        patch = self.normalize(patch)
        future_patch = self.normalize(future_patch)

        lowres_patch = patch[None,:, ::self.factor, ::self.factor]
        lowres_patch =  F.interpolate(lowres_patch, 
                                      size=[patch.shape[1], patch.shape[2]], 
                                      mode='bicubic')[0,-1:]

        # lowres_patch = self.transform(lowres_patch)
        # future_patch = self.transform(future_patch)
        
        # direction = random.choice(['horizontal', 'vertical'])
        # lowres_patch, patch, future_patch = self.transform(lowres_patch, patch, future_patch,direction)

        return lowres_patch, patch, future_patch, torch.tensor(Reynolds_number/40000.).unsqueeze(0)


    def __len__(self):
        #return 100
        return (self.data_shape[0] - self.num_pred_steps - self.cond_snapshots)*self.oversampling +1

    def undo_norm(self,x):
        return x*self.std + self.mean

    def normalize(self,x):
        return (x-self.mean)/self.std

    
    @abstractmethod
    def open_hdf5(self):
        pass

    @abstractmethod
    def get_norm(self):
        pass


    
class NSKT(DataLoader):
    def __init__(self,
                 factor,
                 num_pred_steps=1,
                 patch_size=256,
                 stride = 2,
                 train=True,
                 scratch_dir='./',
                 oversampling = 40,
                 ):
        if train:
            seed = '2150'
        else:
            seed = '3407'            
        self.paths = [os.path.join(scratch_dir,f'2000_2048_2048_seed_{seed}.h5'),
                      os.path.join(scratch_dir,f'4000_2048_2048_seed_{seed}.h5'),
                      os.path.join(scratch_dir,f'8000_2048_2048_seed_{seed}.h5'),
                      os.path.join(scratch_dir,f'16000_2048_2048_seed_{seed}.h5'),
                      os.path.join(scratch_dir,f'32000_2048_2048_seed_{seed}.h5'),
                      ]

        self.RN = [2000,4000,8000,16000,32000]
        super().__init__(factor,num_pred_steps,patch_size,stride,train,scratch_dir,oversampling)
        
    def open_hdf5(self):
        return [h5py.File(path, 'r')['w'] for path in self.paths]

    def get_norm(self):
        return  0.0, 5.4574137
        
    
class E5(DataLoader):
    def __init__(self,
                 factor,
                 num_pred_steps=1,
                 patch_size=256,
                 stride = 2,
                 train=True,
                 scratch_dir='./',
                 oversampling = 40):


        if train:
            self.paths = [os.path.join(scratch_dir,f'{year}.h5') for year in range(1979,2016)]
            self.RN = np.zeros(len(self.paths))
        else:
            self.paths = [os.path.join(scratch_dir,'2016.h5'),os.path.join(scratch_dir,'2017.h5')]
            self.RN = [0.0,0.0]


        super().__init__(factor,num_pred_steps,patch_size,stride,train,scratch_dir,oversampling)
    
    def open_hdf5(self):
        return [h5py.File(path, 'r')['fields'] for path in self.paths]

    def get_norm(self):
        return  278.06824, 21.676298


class Simple(DataLoader):
    def __init__(self,
                 factor,
                 num_pred_steps=1,
                 patch_size=256,
                 stride = 256,
                 train=True,
                 scratch_dir='./',
                 oversampling = 1
                 ):


        if train:
            self.paths = [os.path.join(scratch_dir,'ns_incomp_forced_res256_time10.steps500_visc0.07_reynolds100_wave2_maxvelo7.0_seed0.h5')]
            self.RN = [100.0]
        else:
            self.paths = [os.path.join(scratch_dir,'ns_incomp_forced_res256_time10.steps500_visc0.07_reynolds100_wave2_maxvelo7.0_seed1.h5')]            
            self.RN = [100.0]

        super().__init__(factor,num_pred_steps,patch_size,stride,train,scratch_dir,oversampling)

    def open_hdf5(self):
        return [h5py.File(path, 'r')['tasks']['vorticity'] for path in self.paths]


class EvalLoader(torch.utils.data.Dataset, ABC):
    def __init__(self,
                 factor,
                 step = 0,
                 patch_size=256, 
                 stride = 256,                 
                 horizon = 30,
                 Reynolds_number = 16000,
                 scratch_dir='./',
                 superres = False,
                 shift_factor = 1, #Skip initial snapshot for forecasting
                 skip_factor = 8,  #Avoid overlaping
                 cond_snapshots = 2,
                 ):


        self.factor = factor
        self.patch_size = patch_size
        self.stride = stride
        self.step = step
        self.Reynolds_number = Reynolds_number
        self.horizon = horizon
        self.superres = superres
        self.shift_factor = shift_factor
        self.skip_factor = skip_factor
        self.cond_snapshots = cond_snapshots

        assert Reynolds_number in self.files_dict, "ERROR: Reynolds number not present in evaluation datasets"
        self.file = self.files_dict[Reynolds_number]
        self.dataset = self.open_hdf5()
        self.data_shape = self.dataset.shape
        self.mean, self.std = self.get_norm()

        self.max_row = (self.data_shape[1] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[2] - self.patch_size) // self.stride + 1    

        self.num_patches_per_image = ((self.data_shape[1] - self.patch_size) // self.stride + 1) * \
                                     ((self.data_shape[2] - self.patch_size) // self.stride + 1)
        
    def __len__(self):
        return  self.num_patches_per_image  * 70 #30000 #self.length
        
    
    @abstractmethod
    def open_hdf5(self):
        pass

    @abstractmethod
    def get_norm(self):
        pass

    def undo_norm(self,x):
        return x*self.std + self.mean

    def normalize(self,x):
        return (x-self.mean)/self.std

    
    def __getitem__(self, index):
        if not hasattr(self, 'dataset'):
            self.open_hdf5()

        # deterministic
        num_patches_per_row = (self.data_shape[2] - self.patch_size) // self.stride + 1

        if self.superres:
            snapshot_idx = (index // self.num_patches_per_image) * self.skip_factor
        else:
            snapshot_idx = (index // self.num_patches_per_image) * self.skip_factor + self.shift_factor
            
        patch_idx = index % self.num_patches_per_image
        
        patch_row = (patch_idx // num_patches_per_row) * self.stride
        patch_col = (patch_idx % num_patches_per_row) * self.stride   
        patch = torch.from_numpy(self.dataset[snapshot_idx - self.cond_snapshots + 1 : snapshot_idx + 1, patch_row:(patch_row + self.patch_size), patch_col:(patch_col + self.patch_size)]).float()
        patch = self.normalize(patch)


        lowres_patch = patch[None,:, ::self.factor, ::self.factor]
        lowres_patch =  F.interpolate(lowres_patch, 
                                      size=[patch.shape[1], patch.shape[2]], 
                                      mode='bicubic')[0,-1:]
        
        forecast = []
        for i in range(1,self.horizon):
            forecast.append(
                self.normalize(
                torch.from_numpy(self.dataset[snapshot_idx + (self.step*i), patch_row:(patch_row + self.patch_size),
                                              patch_col:(patch_col + self.patch_size)]).float().unsqueeze(0))
            )
        

        return lowres_patch, patch, forecast, torch.tensor(self.Reynolds_number/40000.).unsqueeze(0)

    def __len__(self):
        return  self.num_patches_per_image  * 70 #30000 #self.length

        

class NSKT_eval(EvalLoader):
    def __init__(self,
                 factor,
                 step = 0,
                 patch_size=256, 
                 stride = 256,                 
                 horizon = 30,
                 Reynolds_number = 16000,
                 scratch_dir='./',
                 superres = False,
                 shift_factor = 600,
                 skip_factor = 8,
                 cond_snapshots = 2,
                 ):


        self.files_dict = {
            600:os.path.join(scratch_dir,'600_2048_2048_seed_3407.h5'),
            1000:os.path.join(scratch_dir,'1000_2048_2048_seed_3407.h5'),
            2000:os.path.join(scratch_dir,'2000_2048_2048_seed_3407.h5'),
            4000:os.path.join(scratch_dir,'4000_2048_2048_seed_3407.h5'),
            8000:os.path.join(scratch_dir, '8000_2048_2048_seed_3407.h5'),
            12000:os.path.join(scratch_dir,'12000_2048_2048_seed_3407.h5'),
            16000:os.path.join(scratch_dir,'16000_2048_2048_seed_3407.h5'),
            24000:os.path.join(scratch_dir,'24000_2048_2048_seed_3407.h5'),  
            32000:os.path.join(scratch_dir,'32000_2048_2048_seed_3407.h5'),
            36000:os.path.join(scratch_dir,'36000_2048_2048_seed_3407.h5'),
        }
        
        super().__init__(factor,step,patch_size,stride,horizon,
                         Reynolds_number,scratch_dir,superres,
                         shift_factor,skip_factor,cond_snapshots)

    
    def open_hdf5(self):
        return h5py.File(self.file, 'r')['w']

    def get_norm(self):
        return  0.0, 5.4574137 


class E5_eval(EvalLoader):
    def __init__(self, factor, 
                 patch_size=256, 
                 stride = 256,
                 step = 0,
                 horizon = 30,
                 Reynolds_number = 1,
                 scratch_dir='./',
                 superres = False,
                 shift_factor = 1,
                 skip_factor = 1,
                 cond_snapshots = 2,
                 ):

        self.files_dict = {
            0:os.path.join(scratch_dir,'2016.h5'),
        }

        super().__init__(factor,step,patch_size,stride,horizon,
                         Reynolds_number,scratch_dir,superres,
                         shift_factor,skip_factor,cond_snapshots)
    
    def open_hdf5(self):
        return h5py.File(self.file, 'r')['fields'][:,2]

    def get_norm(self):
        return  278.06824, 21.676298


class Simple_eval(EvalLoader):
    def __init__(self, factor, 
                 patch_size=256, 
                 stride = 256,
                 step = 0,
                 horizon = 30,
                 Reynolds_number = 1,
                 scratch_dir='./',
                 superres = False,
                 shift_factor = 0,
                 skip_factor = 1,
                 cond_snapshots = 2,
                 ):

        self.files_dict = {
            100:os.path.join(scratch_dir,'ns_incomp_forced_res256_time10.steps500_visc0.07_reynolds100_wave2_maxvelo7.0_seed2.h5'),
            1000:os.path.join(scratch_dir,'ns_incomp_forced_res256_time10.steps500_visc0.007_reynolds1000_wave6_maxvelo7.0_seed0.h5'),
        }

        super().__init__(factor,step,patch_size,stride,horizon,
                         Reynolds_number,scratch_dir,superres,
                         shift_factor,skip_factor,cond_snapshots)
            
    def open_hdf5(self):
        return h5py.File(self.file, 'r')['tasks']['vorticity']