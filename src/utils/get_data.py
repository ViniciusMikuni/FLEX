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


    
class PatchDataset(torch.utils.data.Dataset, ABC):
    """
    SAFE for num_workers > 0 (lazy HDF-5 handles).
    """
    def __init__(self, factor=8, num_pred_steps=1, patch_size=256, stride=64,
                 train=True, oversampling=1, cond_snapshots=2):
        self.factor          = factor
        self.num_pred_steps  = num_pred_steps
        self.train           = train
        self.oversampling    = oversampling
        self.patch_size      = patch_size
        self.stride          = stride
        self.cond_snapshots  = cond_snapshots

        # defer heavy work:
        self.paths = self.build_file_list()
        self.RN    = self.reynolds_numbers()

        # discover data shape from a *temporary* handle
        with h5py.File(self.paths[0], "r") as f:
            self.data_shape = f["w"].shape

        self.max_row = (self.data_shape[-2] - self.patch_size) // self.stride + 1
        self.max_col = (self.data_shape[-1] - self.patch_size) // self.stride + 1
        self.mean, self.std = self.get_norm()

        self._datasets = None          # will hold per-process handles

    # ------------------------------------------------------------------ #
    # ---------- abstract helpers to be implemented by subclass --------- #
    
    @abstractmethod
    def build_file_list(self): ...

    @abstractmethod
    def reynolds_numbers(self): ...

    @abstractmethod
    def get_norm(self): ...

    # ---------- lazy opener ------------------------------------------- #
    def _ensure_open(self):
        if self._datasets is None:     # first touch *in this process*
            self._datasets = [h5py.File(p, "r", libver="latest", swmr=True)["w"]
                              for p in self.paths]

    # ------------------------------------------------------------------ #
    def __len__(self):
        #return ((self.data_shape[0] - self.num_pred_steps - self.cond_snapshots)
        #        * self.oversampling + 1)
    
        return 64

    def normalize(self, x):   return (x - self.mean) / self.std
    def undo_norm(self, x):   return x * self.std + self.mean

    # ------------------------------------------------------------------ #
    def __getitem__(self, index):
        self._ensure_open()
        index = index // self.oversampling + self.cond_snapshots - 1

        row = np.random.randint(0, self.max_row) * self.stride
        col = np.random.randint(0, self.max_col) * self.stride
        ds_id = np.random.randint(0, len(self.paths))

        RN       = self.RN[ds_id]
        dataset  = self._datasets[ds_id]

        if len(self.data_shape) == 4:
            patch = dataset[index - self.cond_snapshots + 1 : index + 1,
                            2, row:row+self.patch_size, col:col+self.patch_size]
            future = dataset[index + self.num_pred_steps,
                             2, row:row+self.patch_size, col:col+self.patch_size][None]
        else:
            patch = dataset[index - self.cond_snapshots + 1 : index + 1,
                            row:row+self.patch_size, col:col+self.patch_size]
            future = dataset[index + self.num_pred_steps,
                             row:row+self.patch_size, col:col+self.patch_size][None]

        patch  = torch.from_numpy(patch).float()
        future = torch.from_numpy(future).float()

        patch  = self.normalize(patch)
        future = self.normalize(future)

        lowres = patch[None, :, ::self.factor, ::self.factor]
        lowres = F.interpolate(lowres, size=patch.shape[1:], mode="bicubic")[0, -1:]

        return (lowres,
                patch,
                future,
                torch.tensor(RN / 40000.).unsqueeze(0))


        
class NSKT(PatchDataset):
    def __init__(self, factor, num_pred_steps=1, patch_size=256, stride=2,
                 train=True, scratch_dir="./", oversampling=40):
        self.scratch_dir = scratch_dir
        self.seed = "2150" if train else "3407"
        super().__init__(factor, num_pred_steps, patch_size, stride,
                         train, oversampling)

    def build_file_list(self):
        seeds = [2000, 4000, 8000, 16000, 32000]
        return [os.path.join(self.scratch_dir,
                             f"{rn}_2048_2048_seed_{self.seed}.h5")
                for rn in seeds]

    def reynolds_numbers(self):
        return [2000, 4000, 8000, 16000, 32000]

    def get_norm(self):
        return 0.0, 5.4574137




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

