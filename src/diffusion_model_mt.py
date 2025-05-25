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
            forecast_encoder: nn.Module,
            n_T: int,
            prediction_type: str,
            sample_loss = False,
            clip_loss = False,
            criterion: nn.Module = nn.L1Loss(reduction='none'),
            use_weight = False,
            sigmoid_shift = -1,
            logsnr_shift = 16.,
            
    ) -> None:
        super(DiffusionModel, self).__init__()

        self.sample_loss = sample_loss
        self.clip_loss = clip_loss
        self.prediction_type = prediction_type
        assert self.prediction_type in ['v','eps','x'], "ERROR: Prediction not supported. Options are v, eps, x"
        
        self.encoder = encoder
        self.decoder = decoder
        self.superres_encoder = superres_encoder
        self.forecast_encoder = forecast_encoder
                
        self.n_T = n_T
        self.criterion = criterion
        self.use_weight = use_weight
        self.sigmoid_shift = sigmoid_shift
        self.logsnr_shift = logsnr_shift

    def forward(self,  
                lowres_snapshots: torch.Tensor,
                snapshots: torch.Tensor,
                future_snapshots: torch.Tensor,
                fluid_condition:torch.Tensor) -> torch.Tensor:

                
        residual_snapshots_SR = snapshots[:,-1:] - lowres_snapshots
        residual_snapshots_FC = future_snapshots - snapshots[:,-1:]
        
        if self.sample_loss:
            _ts = torch.ones((residual_snapshots_FC.shape[0],)).to(residual_snapshots_FC.device)
                
        else:              
            _ts = torch.rand(size=(residual_snapshots_FC.shape[0],)).to(residual_snapshots_FC.device)
                
        eps = torch.randn_like(residual_snapshots_FC)  # eps ~ N(0, 1)
        logsnr, alpha, sigma = get_logsnr_alpha_sigma(_ts,shift=self.logsnr_shift)
            
            
        residual_snapshots_t_SR = alpha * residual_snapshots_SR + eps * sigma        
        residual_snapshots_t_FC = alpha * residual_snapshots_FC + eps * sigma

        if self.sample_loss:
            x_pred_SR = self.sample(residual_snapshots_t_SR.shape[0],
                                    (1, residual_snapshots_t_SR.shape[2],residual_snapshots_t_SR.shape[3]),
                                    lowres_snapshots, fluid_condition,
                                    residual_snapshots_t_SR.device,
                                    superres=True)
            
            x_pred_FC = self.sample(residual_snapshots_t_FC.shape[0],
                                    (1, residual_snapshots_t_FC.shape[2],
                                     residual_snapshots_t_FC.shape[3]),
                                    snapshots, fluid_condition, 
                                    residual_snapshots_t_FC.device)


            predicted = torch.cat([x_pred_SR,x_pred_FC])
            target = torch.cat([snapshots[:,-1:],future_snapshots])
            loss_clip = 0
            w = 1.0
            
        else:

            #Task-specific heads
            head_SR, skips_SR = self.superres_encoder(lowres_snapshots,fluid_condition = fluid_condition)                            
            head_FC, skips_FC = self.forecast_encoder(snapshots,fluid_condition = fluid_condition)
            skip_head = []
            for skip_SR, skip_FC in zip(skips_SR,skips_FC):
                skip_head.append(torch.cat([skip_SR,skip_FC]))


            #General model
            residual_snapshots_t = torch.cat([residual_snapshots_t_SR,residual_snapshots_t_FC],0)
            predicted, skip_connections = self.encoder(residual_snapshots_t,
                                                       torch.cat([_ts,_ts]),
                                                       fluid_condition = torch.cat([fluid_condition,fluid_condition],0),
                                                       cond_skips = skip_head)

            if self.clip_loss:
                #Add clip loss to align the bottlenecks
                pred1,pred2=torch.split(predicted,predicted.shape[0]//2,0)
                loss_clip = CLIPLoss(pred1,pred2)
            else:
                loss_clip = 0.0

                

            predicted = self.decoder(predicted,skip_connections,
                                     torch.cat([head_SR,head_FC]),skip_head,
                                     torch.cat([_ts,_ts]),
                                     fluid_condition = torch.cat([fluid_condition,fluid_condition],0))

            w = torch.ones_like(logsnr)
            # Different predictions schemes
            if self.prediction_type == 'x':
                target = torch.cat([residual_snapshots_SR,residual_snapshots_FC],0)
            elif self.prediction_type == 'eps':
                #Target is eps, but output is v
                if self.use_weight:
                    w = torch.sigmoid(self.sigmoid_shift - logsnr)                    
                pred_SR,pred_FC=torch.split(predicted,predicted.shape[0]//2,0)
                predicted = torch.cat([alpha * pred_SR + sigma * residual_snapshots_t_SR,
                                       alpha * pred_FC + sigma * residual_snapshots_t_FC],0)
                target = torch.cat([eps,eps],0)
            elif self.prediction_type == 'v':
                if self.use_weight:
                    expmb = torch.exp(-torch.tensor(self.sigmoid_shift))
                    w = 1.0/(expmb - torch.sigmoid(-logsnr)*torch.expm1(-torch.tensor(self.sigmoid_shift)))
                target = torch.cat([alpha * eps - sigma * residual_snapshots_SR,
                                    alpha * eps - sigma * residual_snapshots_FC],0)

            w = torch.cat([w,w],0)


            #aux_loss_superres  = sum(self.superres_encoder.extra_losses) if self.superres_encoder.extra_losses else 0
            #aux_loss_forecast  = sum(self.forecast_encoder.extra_losses) if self.forecast_encoder.extra_losses else 0
            #aux_loss_general  = sum(self.encoder.extra_losses) if self.encoder.extra_losses else 0


        return torch.mean(w*self.criterion(predicted, target)) + loss_clip #+ aux_loss_general + aux_loss_forecast + aux_loss_superres


    #@torch.compile
    def sample(self, n_sample: int, size,               
               conditioning_snapshots: torch.Tensor,
               fluid_condition: torch.Tensor,
               device='cuda',superres = False, snapshots_i = None,) -> torch.Tensor:
        
        if snapshots_i is None:
           snapshots_i = torch.randn(n_sample, *size).to(device)  # x_T ~ N(0, 1)

        conditional = conditioning_snapshots.to(device)
        if superres:
            model_head = self.superres_encoder
        else:            
            model_head = self.forecast_encoder

        

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



def CLIPLoss(emb1,emb2,temperature=0.01):
    #Flatten the inputs with take mean
    B, C, H, W = emb1.shape
    emb1 = emb1.view(B,-1)
    emb2 = emb2.view(B,-1)

    emb1 = F.normalize(emb1, p=2, dim=1)  # Normalize along channel dimension
    emb2 = F.normalize(emb2, p=2, dim=1)    # N
        
    # Calculating the Loss
    logits = (emb1 @ emb2.T) / temperature
    emb1_similarity = emb1 @ emb1.T
    emb2_similarity = emb2 @ emb2.T
    #targets = F.softmax((emb1_similarity + emb2_similarity) / 2 * temperature, dim=-1)
    targets = torch.arange(B).to(emb1.device)
    
    emb2_loss = F.cross_entropy(logits, targets, reduction='none')
    emb1_loss = F.cross_entropy(logits.T, targets, reduction='none')
    loss =  (emb1_loss + emb2_loss) / 2.0 # shape: (batch_size)
    return loss.mean()

    
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

    
