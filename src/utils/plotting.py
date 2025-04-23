#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 14:02:58 2024

@author: ben
"""

import numpy as np
import matplotlib.pyplot as plt
import cmocean

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


def plot_samples(samples, conditioning_snapshots, targets, PATH, epoch):
    
        samples = samples.cpu().detach().numpy()
        conditioning_snapshots = conditioning_snapshots.cpu().detach().numpy()        
        targets = targets.cpu().detach().numpy()        
    
        nrow = 3; ncol = 4;
        f, axarr = plt.subplots(nrow, ncol, figsize=(12, 8))

        for i in range(ncol): 
           axarr[0,i].imshow(conditioning_snapshots[i,0,:,:], cmap=cmocean.cm.balance)
           axarr[0,i].set_xticks([])
           axarr[0,i].set_yticks([])
           axarr[0,i].title.set_text("LR input")  



        for i in range(ncol):
           error = np.linalg.norm(samples[i,0,:,:] - targets[i,0,:,:]) / np.linalg.norm(targets[i,0,:,:])
           axarr[1,i].imshow(samples[i,0,:,:], cmap=cmocean.cm.balance)
           axarr[1,i].set_xticks([])
           axarr[1,i].set_yticks([])
           axarr[1,i].title.set_text(f"Super-resolved (RFNE={error:.3f})")  
           
           # Draw zoom in 
           zoom_in_factor = 2
           upscale_factor = 7.5
           ec = ".2"
           zoom_loc_x = (1000,1400)
           zoom_loc_y = (500,100)
           axins = zoomed_inset_axes(axarr[1,i], zoom_in_factor, loc='lower left')    
           axins.imshow(samples[i,0,:,:], cmap=cmocean.cm.balance)
           axins.set_xlim(tuple(val // upscale_factor for val in zoom_loc_x))
           axins.set_ylim(tuple(val // upscale_factor for val in zoom_loc_y))
           plt.xticks(visible=False)
           plt.yticks(visible=False)
           _patch, pp1, pp2 = mark_inset(axarr[1,i], axins, loc1=4, loc2=2, fc="none", ec=ec, lw=1.0, color='black') 
           #pp1.loc1, pp1.loc2 = 2, 3  # inset corner 2 to origin corner 3 (would expect 2)
           #pp2.loc1, pp2.loc2 = 4, 1  # inset corner 4 to origin corner 1 (would expect 4)
           pp1.loc1, pp1.loc2 = 4, 1  # inset corner 1 to origin corner 4 (would expect 1)
           pp2.loc1, pp2.loc2 = 2, 2  # inset corner 3 to origin corner 2 (would expect 3)
           plt.draw()               

        for i in range(ncol): 
           axarr[2,i].imshow(targets[i,0,:,:], cmap=cmocean.cm.balance)
           axarr[2,i].set_xticks([])
           axarr[2,i].set_yticks([])
           axarr[2,i].title.set_text("HR Ground Truth")                 

           # Draw zoom in 
           zoom_in_factor = 2
           upscale_factor = 7.5
           ec = ".2"
           zoom_loc_x = (1000,1400)
           zoom_loc_y = (500,100)
           axins = zoomed_inset_axes(axarr[2,i], zoom_in_factor, loc='lower left')    
           axins.imshow(targets[i,0,:,:], cmap=cmocean.cm.balance)
           axins.set_xlim(tuple(val // upscale_factor for val in zoom_loc_x))
           axins.set_ylim(tuple(val // upscale_factor for val in zoom_loc_y))
           plt.xticks(visible=False)
           plt.yticks(visible=False)
           _patch, pp1, pp2 = mark_inset(axarr[2,i], axins, loc1=4, loc2=2, fc="none", ec=ec, lw=1.0, color='black') 
           #pp1.loc1, pp1.loc2 = 2, 3  # inset corner 2 to origin corner 3 (would expect 2)
           #pp2.loc1, pp2.loc2 = 4, 1  # inset corner 4 to origin corner 1 (would expect 4)
           pp1.loc1, pp1.loc2 = 4, 1  # inset corner 1 to origin corner 4 (would expect 1)
           pp2.loc1, pp2.loc2 = 2, 2  # inset corner 3 to origin corner 2 (would expect 3)
           plt.draw() 
    

        plt.tight_layout()
        plt.savefig(PATH + "/ddpm_sample_"+ str(epoch) + ".png")
        plt.close()
        
        
# testing ---        
#train_set = NSTK_SR(path='../data/16000_2048_2048_seed_3407_w.h5', factor = 8)  # load your dataset
#from torch.utils.data import Dataset, DataLoader
#dataloader =  DataLoader(train_set, batch_size=32, shuffle=True)
    
#conditioning_snapshots, targets = next(iter(dataloader))
#samples = targets

#plot_samples(samples, conditioning_snapshots, targets, PATH='', epoch=1)
        