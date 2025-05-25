"""
Created on Fri Jul  23, 2024

@author: ben
"""

import numpy as np


import os,sys, time
import wandb
import torch
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, barrier, get_rank, is_initialized, all_reduce, get_world_size
from torch_ema import ExponentialMovingAverage
#import torch.cuda.amp as amp

from torch.optim import lr_scheduler

from src.backbones.unet import UNet
from src.backbones.flex import FLEX
#from src.backbones.uhybrid_moe import UViTHybridMoE
from src.backbones.uvit import UViT
from src.diffusion_model import DiffusionModel
from src.utils.get_data import NSKT #,E5, Simple
from src.utils.plotting import plot_samples
from src.utils.lion import Lion

from diffusers.optimization import get_cosine_schedule_with_warmup as scheduler


def ddp_setup(local_rank, world_size):
    """
    Args:
        rank: Unique identifixer of each process
        world_size: Total number of processes
    """
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "3522"
        init_process_group(backend="nccl", rank=local_rank, world_size=world_size)
        rank = local_rank
    else:
        init_process_group(backend="nccl", 
                           init_method='env://')
        #overwrite variables with correct values from env
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = get_rank()

    torch.cuda.set_device(local_rank)
    torch.backends.cudnn.benchmark = True
    return local_rank, rank


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            val_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            gpu_id: int,
            local_gpu_id: int,
            run: wandb,
            epochs: int,
            run_name: str,
            scratch_dir: str,
            ema_val = 0.999,
            clip_value = 1.0,
            fine_tune = False,
            dataset = 'nskt',
            sampling_freq = 10,
            use_amp= True,
            undo_norm = None,
    ) -> None:
        self.gpu_id = gpu_id
        self.local_gpu_id = local_gpu_id
        self.model = model.to(local_gpu_id)
        self.ema = ExponentialMovingAverage(model.parameters(), decay=ema_val)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.clip_value = clip_value
        self.model = DDP(model, device_ids=[local_gpu_id],
                         #find_unused_parameters=True
                         )
        self.run = run
        self.run_name = run_name
        self.fine_tune = fine_tune
        if self.fine_tune:
            self.run_name += '_fine_tune'
            # Can freeze some parameter based on the fine-tuning strategy
            # for param in self.model.module.encoder.parameters():
            #     param.requires_grad = False                
            # for param in self.model.module.decoder.parameters():
            #     param.requires_grad = False
                
        self.dataset = dataset
        self.logs = {}
        self.startEpoch = 0
        self.best_loss = np.inf
        self.max_epochs = epochs
        self.sampling_freq = sampling_freq
        self.undo_norm = undo_norm
        self.checkpoint_dir = os.path.join(scratch_dir,'checkpoints')
        self.checkpoint_path = os.path.join(self.checkpoint_dir,f"checkpoint_{self.dataset}_{self.run_name}.pt")
        self.use_amp = use_amp
        if self.use_amp:
            self.gscaler = torch.GradScaler("cuda")
        
        self.lr_scheduler = scheduler(
            optimizer=self.optimizer,
            num_warmup_steps=len(self.train_data), #5, # we need only a very shot warmup phase for our data
            num_training_steps=(len(self.train_data) * self.max_epochs),
        )
                
        if os.path.isfile(self.checkpoint_path):
            if self.gpu_id ==0: print(f"Loading checkpoint from {self.checkpoint_path}")
            self._restore_checkpoint(self.checkpoint_path)
        elif self.fine_tune:
            pretrained_checkpoint = self.checkpoint_path.replace(f"_{self.dataset}_","_nskt_").replace("_fine_tune","")
            if self.gpu_id ==0: print(f"Loading pretrained checkpoint from {pretrained_checkpoint}")
            self._restore_checkpoint(pretrained_checkpoint,restore_all=False)

    def train_one_epoch(self):
        tr_time = 0
        self.model.train()
        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.local_gpu_id)
        self.logs['train_loss'] = logs_buff[0].view(-1)

        tr_start = time.time()
        for data  in self.train_data:
            data = [x.to(self.local_gpu_id,dtype=torch.float) for x in data]
            data_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)
            
            if self.use_amp:
                with torch.autocast("cuda"):
                    loss = self.model(*data)
            else:
                    loss = self.model(*data)


            if self.use_amp:
                self.gscaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                self.gscaler.step(self.optimizer)
                scale = self.gscaler.get_scale()
                self.gscaler.update()
                skip_lr_sched = (scale != self.gscaler.get_scale())
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_value)
                self.optimizer.step()
                skip_lr_sched = False

            if not skip_lr_sched:
                self.lr_scheduler.step()
            self.ema.update()
 
            # add all the minibatch losses
            self.logs['train_loss'] += loss.detach()
            tr_time += time.time() - tr_start
            
        self.logs['train_loss'] /= len(self.train_data)

        logs_to_reduce = ['train_loss']
        if is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/get_world_size())
                
        return tr_time



    def val_one_epoch(self):
        val_time = time.time()
        self.model.eval()
        
        # buffers for logs
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.local_gpu_id)
        self.logs['val_loss'] = logs_buff[0].view(-1)

        with torch.no_grad():
            for data in self.val_data:
                data_start = time.time()
                data = [x.to(self.local_gpu_id,dtype=torch.float) for x in data]                
                loss = self.model(*data)
                        
                # add all the minibatch losses
                self.logs['val_loss'] += loss.detach()
            
        self.logs['val_loss'] /= len(self.val_data)

        logs_to_reduce = ['val_loss']
        if is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/get_world_size())
                
        return time.time() - val_time


    def _save_checkpoint(self, epoch,PATH):
        save_dict = {
            'encoder': self.model.module.encoder.state_dict(),
            'superres_encoder': self.model.module.superres_encoder.state_dict(),
            'forecast_encoder': self.model.module.forecast_encoder.state_dict(),
            'decoder': self.model.module.decoder.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch':epoch,
            'loss':self.best_loss,
            'sched': self.lr_scheduler.state_dict(),
        }

        if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)
        
        torch.save(save_dict, PATH)
        print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")

    def _restore_checkpoint(self,PATH,restore_all = True):
        checkpoint = torch.load(PATH, map_location='cuda:{}'.format(self.local_gpu_id))

        self.model.module.encoder.load_state_dict(checkpoint["encoder"])
        self.model.module.superres_encoder.load_state_dict(checkpoint["superres_encoder"])
        self.model.module.forecast_encoder.load_state_dict(checkpoint["forecast_encoder"])
        self.model.module.decoder.load_state_dict(checkpoint["decoder"])
        self.ema.load_state_dict(checkpoint["ema"])

        if restore_all:
            self.startEpoch = checkpoint['epoch'] + 1
            if 'loss' in checkpoint:
                self.best_loss = checkpoint['loss']                
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'sched' in checkpoint:
                self.lr_scheduler.load_state_dict(checkpoint['sched'])


    def _generate_samples(self, epoch, ibreak = 5):
        logs_buff = torch.zeros((1), dtype=torch.float32, device=self.local_gpu_id)
        self.logs['RFNE'] = logs_buff[0].view(-1)

        PATH = "./train_samples_" + self.run_name
        if self.gpu_id == 0:            
            if not os.path.exists(PATH):
                os.makedirs(PATH)
                
        

        with self.ema.average_parameters():
            self.model.eval()
            with torch.no_grad():                    
                for i, data in enumerate(self.val_data):
                    data = [x.to(self.local_gpu_id,dtype=torch.float) for x in data]
                    lowres_snapshots, snapshots, future_snapshots, Reynolds_number = data
                    samples = self.model.module.sample(snapshots.shape[0],
                                                       (1, snapshots.shape[2], snapshots.shape[3]), 
                                                       lowres_snapshots, Reynolds_number,
                                                       lowres_snapshots.device,superres=True)
                    predictions = self.undo_norm(samples[:,0])
                    snapshots = self.undo_norm(snapshots[:,0])
                    rfne = torch.linalg.norm(predictions - snapshots)/torch.linalg.norm(snapshots)
                    self.logs['RFNE'] += rfne.mean().detach()
                    if i>ibreak: break
                self.logs['RFNE'] /= float(ibreak)
            
        # plot_samples(samples, lowres_snapshots, snapshots[:,-1,None], PATH, epoch)
        # print(f"Epoch {epoch} | Generated samples saved at {PATH}")

        logs_to_reduce = ['RFNE']
        if is_initialized(): # reduce the logs across multiple GPUs
            for key in logs_to_reduce:
                all_reduce(self.logs[key].detach())
                self.logs[key] = float(self.logs[key]/get_world_size())

        
    
    def train(self):
        for epoch in range(self.startEpoch,self.max_epochs):
            if is_initialized():
                self.train_data.sampler.set_epoch(epoch)
            
            start = time.time()
            tr_time  = self.train_one_epoch()
            val_time = self.val_one_epoch()

            if (epoch + 1) % self.sampling_freq == 0:
                self._generate_samples(epoch+1)

            
            if self.gpu_id == 0: 
                print("Epoch {} | loss {} | val loss {} | learning rate {}".format(epoch,self.logs['train_loss'],self.logs['val_loss'],self.lr_scheduler.get_last_lr()))
                print('Time taken for epoch {} is {} sec'.format(epoch, time.time()-start))
                if self.run is not None:
                    self.run.log({"val_loss": self.logs['val_loss']})
                    self.run.log({"train_loss": self.logs['train_loss']})
                    if 'RFNE' in self.logs:
                        self.run.log({"RFNE": self.logs['RFNE']})
                    
                    
            if self.gpu_id == 0 and self.logs['val_loss'] < self.best_loss:
                print("replacing best checkpoint ...")
                self.best_loss = self.logs['val_loss']
                self._save_checkpoint(epoch+1,self.checkpoint_path)



                  
def load_train_objs(args):
    if  args.dataset == 'nskt':
        train_set = NSKT(factor=args.superres_factor, num_pred_steps=args.forecast_steps,
                         scratch_dir=args.data_dir)
        val_set = NSKT(factor=args.superres_factor, num_pred_steps=args.forecast_steps,train=False,
                       scratch_dir=args.data_dir)

    
    if args.model == 'unet':
        backbone = UNet
    elif args.model == 'uvit':
        backbone = UViT        
    elif args.model == 'flex':
        backbone = FLEX
     
    else:
        print("ERROR: Model not found")
        sys.exit()

        
    encoder,superres_encoder,forecast_encoder, decoder = backbone(image_size=256,
                                                                  in_channels=1,
                                                                  out_channels=1,
                                                                  model_size=args.size,
                                                                  cond_snapshots = args.cond_snapshots)

        
    if args.criterion == 'l1':
        criterion = torch.nn.L1Loss(reduction='none')
    elif args.criterion == 'mse':
        criterion = torch.nn.MSELoss(reduction='none')
    elif args.criterion == 'huber':
        criterion = torch.nn.HuberLoss(reduction='none')
    else:
        raise ValueError("Loss function not supported")
    
    model = DiffusionModel(encoder = encoder.cuda(),
                           decoder = decoder.cuda(),
                           superres_encoder = superres_encoder.cuda(),
                           forecast_encoder = forecast_encoder.cuda(),
                           n_T=args.time_steps, 
                           prediction_type = args.prediction_type, 
                           sample_loss = args.sample_loss,
                           clip_loss = args.clip_loss,
                           criterion = criterion,
                           use_weight = args.sigmoid_weight,
                           sigmoid_shift = args.sigmoid_shift,
                           logsnr_shift = args.logsnr_shift,
                           )
        
    factor = 1
    if args.fine_tune:
        factor = 3.
    if args.optimizer == 'lion':
        optimizer = Lion(model.parameters(), lr=args.learning_rate/factor,betas=(args.lion_b1,args.lion_b2),weight_decay=0.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate/factor,weight_decay=0.0)
    return train_set,val_set, model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=DistributedSampler(dataset),
        #pin_memory=torch.cuda.is_available(),
        pin_memory=True,
        #persistent_workers=True,       # keeps the handles alive
        num_workers=8,
    )


def main(rank: int, world_size: int, epochs: int, batch_size: int, run, args):


    local_rank, rank = ddp_setup(rank, world_size)
    device = torch.cuda.current_device()
    train_data,val_data,  model, optimizer = load_train_objs(args=args)
    undo_norm = train_data.undo_norm
    model = model.to(device)

    if rank == 0:
        #==============================================================================
        # Model summary
        #==============================================================================
        print('**** Setup ****')
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
        print('************')

    
    train_data = prepare_dataloader(train_data, batch_size)
    val_data = prepare_dataloader(val_data, batch_size)
    
    trainer = Trainer(model, train_data,val_data, optimizer,
                      rank,local_rank, run, epochs = epochs,
                      run_name=args.run_name,
                      scratch_dir = args.scratch_dir,
                      fine_tune=args.fine_tune,
                      dataset=args.dataset,
                      use_amp = False,
                      #True if args.model == 'uvit' else False,
                      undo_norm=undo_norm,
                      sampling_freq = args.sampling_freq)
    trainer.train()
    destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='nskt', help="Name of the dataset to train. Options are [nskt,climate,simple]")
    parser.add_argument("--model", type=str, default='uvit', help="Model used as the backbone")
    parser.add_argument("--size", type=str, default='medium', help="Model size. Options are [small, medium, big]")
    parser.add_argument("--data-dir", type=str, default='data/', help="path to data folder")

    #General parameters
    parser.add_argument("--scratch-dir", type=str, default='/', help="Name of the current run.")
    parser.add_argument('--epochs', default=200, type=int, help='Total epochs to train the model')
    parser.add_argument('--sampling-freq', default=5, type=int, help='How often to save a snapshot')
    parser.add_argument('--batch-size', default=8, type=int, help='Input batch size on each device (default: 8)')

    #Tasks
    parser.add_argument('--forecast_steps', default=1, type=int, help='different prediction steps to condition on')
    parser.add_argument('--superres_factor', default=8, type=int, help='upsampling factor')
    parser.add_argument('--cond_snapshots', default=2, type=int, help='Previous snapshots to condition for forecast')
    

    #Optimizer
    parser.add_argument('--optimizer', default="lion", type=str, help='Optimizer to use: supported options are lion and adam')
    parser.add_argument('--learning-rate', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--lion-b1', default=0.90, type=float, help='Lion optimizer beta1')
    parser.add_argument('--lion-b2', default=0.98, type=float, help='Lion optimizer beta1')

    #Diffusion parameters
    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")
    parser.add_argument("--time-steps", type=int, default=2, help="Time steps for sampling")
    parser.add_argument("--sigmoid_weight", action='store_true', default=False, help='Use sigmoid weight')
    parser.add_argument('--sigmoid_shift', default=-1., type=float, help='Shift for sigmoid weight')
    parser.add_argument('--logsnr_shift', default=1., type=float, help='Shift logsnr value')
    parser.add_argument("--criterion", type=str, default='l1', help="Loss function to use. Options are l1 and mse")
    parser.add_argument("--sample_loss", action='store_true', default=False, help='Run the model calling the generation step during training')
    parser.add_argument("--clip_loss", action='store_true', default=False, help='Run the model calling the generation step during training')

    
    parser.add_argument("--multi-node", action='store_true', default=False, help='Use multi node training')
    parser.add_argument("--fine_tune", action='store_true', default=False, help='Fine tune using pretrained model')
    

    args = parser.parse_args()

    # start
    np.random.seed(1)
    print('start')
    
    if args.multi_node:
        def is_master_node():
            return int(os.environ['RANK']) == 0
        
        if is_master_node():
            #mode = "disabled"
            mode = None
            wandb.login()
        else:
            mode = "disabled"
            

        run = wandb.init(
            # Set the project where this run will be logged
            project="FLEX4",
            name=args.run_name,
            mode = mode,
            # Track hyperparameters and run metadata            
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "upsampling factor": args.superres_factor,
            },
        )
            
        
        main(0,1, args.epochs, args.batch_size, run, args)
    else:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project="FLEX5",
            name=args.run_name,
            #mode = 'disabled',
            # Track hyperparameters and run metadata
            config={
                "learning_rate": args.learning_rate,
                "epochs": args.epochs,
                "batch size": args.batch_size,
                "upsampling factor": args.superres_factor,
            },
        )

        world_size = torch.cuda.device_count()
        #Launch processes.
        print('Launching processes...')
        mp.spawn(main, args=(world_size, args.epochs, args.batch_size, run, args), nprocs=world_size)
        #main(0,1, args.epochs, args.batch_size, run, args)



# export MKL_THREADING_LAYER=GNU   # before you run Python
# export NCCL_ALGO=Ring           # or Tree
# export NCCL_P2P_DISABLE=1       # disables NVLink/SHARP fallback paths
# export CUDA_VISIBLE_DEVICES=0,1,2,3; python train.py --run-name flex_v_small --dataset nskt --model flex --size small --data-dir /data/rdl/NSTK/


