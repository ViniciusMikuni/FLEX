.. image:: images/logo.png
    :width: 500px

# Self-contained multiple expert models

## Training a new model for both superresolution and forecasting tasks

Training a new model based on the NSTK or Climate datasets can be performed with the following command:

```bash
python train.py --run-name hybrid_v_small --dataset nskt --model hybrid --size small [--multi-node] --scratch-dir PATH/TO/DATASET
```

Additional options are available for model sizes (small/medium/big), model types (unet,uvit,hybrid), and datasets (nskt/climate).

## Evaluation with error metrics

To evaluate the trained model you can use the evaluation code bellow.

## Forecasting

```bash
python evaluate.py --run-name hybrid_v_small  --Reynolds-number 12000 --batch-size 32  --horizon 10  --diffusion-steps 2 --model hybrid  --ensemb-size 1 --size small
python evaluate.py --run-name hybrid_v_small  --Reynolds-number 0 --batch-size 32  --horizon 10  --diffusion-steps 2 --model hybrid  --ensemb-size 1 --size small --dataset climate
```

## Superresolution

```bash
python evaluate.py --run-name hybrid_v_small  --Reynolds-number 12000 --batch-size 32  --diffusion-steps 2 --model hybrid  --ensemb-size 1 --size small --superres
python evaluate.py --run-name hybrid_v_small  --Reynolds-number 0 --batch-size 32  --diffusion-steps 2 --model hybrid  --ensemb-size 1 --size small --dataset climate --superres
```