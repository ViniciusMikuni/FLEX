"""
Created on Fri Jul  5, 2024

@author: ben
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import torch.nn.functional as F
import copy


class DiffusionModel(nn.Module):
    def __init__(
            self,
            encoder: nn.Module,
            decoder: nn.Module,
            superres_encoder: nn.Module,
            n_T: int,
            prediction_type: str,
            sample_loss = False,
            criterion: nn.Module = nn.L1Loss(reduction='none'),
            use_weight = False,
            sigmoid_shift = -1,
            logsnr_shift = 16.,
            
    ) -> None:
        super(DiffusionModel, self).__init__()

        self.sample_loss = sample_loss
        self.prediction_type = prediction_type
        assert self.prediction_type in ['v','eps','x'], "ERROR: Prediction not supported. Options are v, eps, x"
        
        self.encoder = encoder
        self.decoder = decoder
        self.superres_encoder = superres_encoder
                
        self.n_T = n_T
        self.criterion = criterion
        self.use_weight = use_weight
        self.sigmoid_shift = sigmoid_shift
        self.logsnr_shift = logsnr_shift

    def forward(self,  
                lowres_snapshots: torch.Tensor,
                snapshots: torch.Tensor,
                fluid_condition:torch.Tensor) -> torch.Tensor:

                
        residual_snapshots_SR = snapshots[:,-1:] - lowres_snapshots
        
        if self.sample_loss:
            _ts = torch.ones((residual_snapshots_SR.shape[0],)).to(residual_snapshots_SR.device)
                
        else:              
            _ts = torch.rand(size=(residual_snapshots_SR.shape[0],)).to(residual_snapshots_SR.device)
                
        eps = torch.randn_like(residual_snapshots_SR)  # eps ~ N(0, 1)
        logsnr, alpha, sigma = get_logsnr_alpha_sigma(_ts,shift=self.logsnr_shift)
            
            
        residual_snapshots_t_SR = alpha * residual_snapshots_SR + eps * sigma        


        if self.sample_loss:
            x_pred_SR = self.sample(residual_snapshots_t_SR.shape[0],
                                    (1, residual_snapshots_t_SR.shape[2],residual_snapshots_t_SR.shape[3]),
                                    lowres_snapshots, fluid_condition,
                                    residual_snapshots_t_SR.device,
                                    superres=True)
            

            predicted = x_pred_SR
            target = snapshots[:,-1:]
            w = 1.0
            
        else:

            #Task-specific heads
            head_SR, skips_SR = self.superres_encoder(lowres_snapshots,fluid_condition = fluid_condition)                            


            #General model
            predicted, skip_connections = self.encoder(residual_snapshots_t_SR,
                                                       _ts,
                                                       fluid_condition = fluid_condition,
                                                       cond_skips = skips_SR)

            predicted = self.decoder(predicted, skip_connections,
                                     head_SR, skips_SR, _ts,
                                     fluid_condition = fluid_condition)

            w = torch.ones_like(logsnr)
            # Different predictions schemes
            if self.prediction_type == 'x':
                target = residual_snapshots_SR
            
            elif self.prediction_type == 'eps':
                #Target is eps, but output is v
                if self.use_weight:
                    w = torch.sigmoid(self.sigmoid_shift - logsnr)                    
                pred_SR,pred_FC=torch.split(predicted,predicted.shape[0]//2,0)
                predicted = alpha * pred_SR + sigma * residual_snapshots_t_SR
                target = eps
            
            elif self.prediction_type == 'v':
                if self.use_weight:
                    expmb = torch.exp(-torch.tensor(self.sigmoid_shift))
                    w = 1.0/(expmb - torch.sigmoid(-logsnr)*torch.expm1(-torch.tensor(self.sigmoid_shift)))
                target = alpha * eps - sigma * residual_snapshots_SR

        return torch.mean(w * self.criterion(predicted, target))

    #@torch.compile
    def sample(self, n_sample: int, size,               
               conditioning_snapshots: torch.Tensor,
               fluid_condition: torch.Tensor,
               device='cuda',superres = False, snapshots_i = None,) -> torch.Tensor:
        
        if snapshots_i is None:
           snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        conditional = conditioning_snapshots.to(device)
        model_head = self.superres_encoder           

        

        for time_step in range(self.n_T, 0, -1):
            time = torch.ones((n_sample,) ).to(device) * time_step / self.n_T
            time_ = torch.ones((n_sample,) ).to(device) * (time_step-1) / self.n_T
            logsnr, alpha, sigma = get_logsnr_alpha_sigma(time,shift=self.logsnr_shift)
            logsnr_, alpha_, sigma_ = get_logsnr_alpha_sigma(time_,shift=self.logsnr_shift)
            
            
            pred_head, skip_head = model_head(conditional,fluid_condition = fluid_condition)
            
            pred, skip = self.encoder(snapshots_i,time.to(device),
                                      fluid_condition = fluid_condition,
                                      cond_skips = skip_head)
            
                        
            pred = self.decoder(pred, skip,pred_head,skip_head,
                                time.to(device),
                                fluid_condition = fluid_condition)

            
            if self.prediction_type == 'v':                    
                mean = alpha * snapshots_i - sigma * pred
                eps = pred * alpha + snapshots_i * sigma
                
            elif self.prediction_type == 'x':
                mean = pred
                eps = (alpha * pred - snapshots_i) / sigma
                
            elif self.prediction_type == 'eps':
                mean = alpha * snapshots_i - sigma * pred
                eps = pred * alpha + snapshots_i * sigma
                
            snapshots_i = alpha_ * mean + eps * sigma_

        

        #Replace last prediction with the mean value
        snapshots_i = mean
        if conditional.shape[1] >1:
            conditional = conditional[:,-1,None]
        return snapshots_i + conditional

    
@torch.compile
def logsnr_schedule_cosine(t, logsnr_min=-20., logsnr_max=20., shift = 1.):
    b = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_max)))
    a = torch.atan(torch.exp(-0.5 * torch.tensor(logsnr_min))) - b
    return -2. * torch.log(torch.tan(a * t + b)*shift)

#@torch.compile
def get_logsnr_alpha_sigma(time,shift=16.):
    logsnr = logsnr_schedule_cosine(time,shift=shift)[:,None,None,None]
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return logsnr, alpha, sigma

    
