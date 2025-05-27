"""
Created on Sat Jul 27 13:10:38 2024

@author: ben
"""


import os, sys
import torch


import numpy as np

from src.backbones.unet import UNet
from src.backbones.uvit import UViT
from src.backbones.flex import FLEX
from src.diffusion_model_sr import DiffusionModel

from torch.utils.data import Dataset, DataLoader
from torch_ema import ExponentialMovingAverage
from tqdm import tqdm
from src.utils.plotting import plot_samples
import h5py
import scipy.stats
import matplotlib.pyplot as plt
import cmocean

from PIL import Image
from PIL import ImageDraw,ImageFont

from src.utils.get_data_sr import NSKT_eval



def generate_samples(model,ema,dataset,undo_norm,args):
    PATH = "./train_samples_" + args.run_name
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    if not os.path.exists(PATH+"_std"):
        os.makedirs(PATH+"_std")
    RFNE_error = []
    R2s = []
    i=0
    
    
    with ema.average_parameters():
        model.eval()
        with torch.no_grad():                
            for lowres_snapshots, snapshots, fluid_condition in tqdm(dataset):
                fluid_condition = fluid_condition.to('cuda',dtype=torch.float)

                cond = lowres_snapshots.to('cuda',dtype=torch.float)
                targets = snapshots[:,-1:].to('cuda',dtype=torch.float)

                predictions = []
                for _ in range(args.ensemb_size): 
                    prediction = model.sample(cond.shape[0], 
                                              (1, targets.shape[2], targets.shape[3]),
                                              cond, fluid_condition,'cuda',
                                              superres=args.superres)

                    predictions.append(prediction)

                std = torch.std(undo_norm(torch.stack(predictions)),0)
                predictions = torch.mean(undo_norm(torch.stack(predictions)),0)
                targets = undo_norm(targets)
                error = torch.linalg.norm(predictions[:,0] - targets[:,0])/torch.linalg.norm(targets[:,0])
                RFNE_error.append(error.cpu().detach().numpy())

                if i ==0:
                    #Make some plots
                    PATH = "./train_samples_" + args.run_name
                    plot_samples(predictions, cond, targets, PATH, 0)
                    plot_samples(std, cond, targets, PATH+"_std", 0)
                    samples = {
                        'conditioning_snapshots': cond.cpu().detach().numpy(),
                        'targets': targets.cpu().detach().numpy(),
                        'predictions': predictions.cpu().detach().numpy(),
                        'error': std.cpu().detach().numpy()
                    }

                    if not os.path.exists("./samples"):
                        os.makedirs("./samples")

                    np.save(
                        f'samples/samples_superres_RE_{args.Reynolds_number}_SR_{args.diffusion_steps}_{args.model}_{args.size}_' + str(i+1) + '.npy', samples)
                    print('saved samples')
                i += 1

        avg_RFNE = np.mean(RFNE_error)                    
        print(f'Average RFNE={repr(avg_RFNE)}')



    
    
def main(args):
    if args.dataset == 'nskt':
        dataset = NSKT_eval(factor=args.factor,
                            step = args.step,
                            Reynolds_number = args.Reynolds_number,
                            horizon=args.horizon,
                            scratch_dir=args.data_dir,
                            superres=args.superres,
                            cond_snapshots = 1 if args.superres else args.cond_snapshots)

    undo_norm = dataset.undo_norm

    if args.model == 'unet':
        backbone = UNet
    elif args.model == 'uvit':
        backbone = UViT        
    elif args.model == 'flex':
        backbone = FLEX
    else:
        print("ERROR: Model not found")
        sys.exit()

    encoder,superres_encoder,_, decoder = backbone(image_size=256,
                                                                  in_channels=1,
                                                                  out_channels=1,
                                                                  model_size=args.size,
                                                                  cond_snapshots = args.cond_snapshots)

        
    model = DiffusionModel(encoder = encoder.cuda(),
                           decoder = decoder.cuda(),
                           superres_encoder = superres_encoder.cuda(),
                           n_T=args.diffusion_steps, 
                           prediction_type = args.prediction_type,
                           logsnr_shift = args.logsnr_shift,
                           )
    
    dataset = DataLoader(
        dataset,
        batch_size=args.batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=8
    )

    print(f"Evaluating model {args.run_name}")
    PATH = args.model_path


    checkpoint = torch.load(PATH)
    #optimizer.load_state_dict(checkpoint["optimizer"])
    model.encoder.load_state_dict(checkpoint["encoder"])
    model.superres_encoder.load_state_dict(checkpoint["superres_encoder"])
    model.decoder.load_state_dict(checkpoint["decoder"])
    
    ema = ExponentialMovingAverage(model.parameters(),decay=0.999)
    ema.load_state_dict(checkpoint["ema"])

    # set seed
    seed = 0
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    generate_samples(model,ema,dataset,undo_norm,args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Minimalistic Diffusion Model for Super-resolution')
    
    
    parser.add_argument('--batch-size', default=32, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--ensemb-size', default=1, type=int, help='Number of ensembles per prediction')
    parser.add_argument('--horizon', default=30, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument("--run-name", type=str, default='run1', help="Name of the current run.")
    parser.add_argument("--dataset", type=str, default='nskt', help="Name of the dataset for evaluation.")
    parser.add_argument("--model", type=str, default='flex', help="Model used as the backbone")
    parser.add_argument("--size", type=str, default='small', help="Model size. Options are [small, medium, big]")
    parser.add_argument('--superres', action='store_true', default=False, help='Superresolution')
    parser.add_argument("--data-dir", type=str, default='data/', help="path to data folder")
    parser.add_argument("--model-path", type=str, default='checkpoint_nskt_flex_v_small.pt', help="name of checkpoint file")


    parser.add_argument('--logsnr_shift', default=1., type=float, help='Shift logsnr value')
    parser.add_argument('--Reynolds-number', default=0, type=int, help='Reynolds number')
    parser.add_argument('--factor', default=8, type=int, help='upsampling factor')
    parser.add_argument('--step', default=1, type=int, help='future time steps to predict')
    parser.add_argument('--cond_snapshots', default=2, type=int, help='Previous snapshots to condition for forecast')

    parser.add_argument("--prediction-type", type=str, default='v', help="Quantity to predict during training.")    
    parser.add_argument("--diffusion-steps", type=int, default=2, help="Time steps for sampling")    

    args = parser.parse_args()
    
    main(args)       
   

# export CUDA_VISIBLE_DEVICES=7; python evaluate_sr.py --model-path checkpoints/checkpoint_nskt_flex_sr_v_small.pt  --Reynolds-number 16000 --batch-size 64 --diffusion-steps 2 --model flex --ensemb-size 1 --size small --superres --data-dir /data/rdl/NSTK/