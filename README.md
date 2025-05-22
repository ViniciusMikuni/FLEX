.. -*- mode: rst -*-

.. image:: images/logo.png
    :width: 300px

====================================
FLEX: A Backbone for Diffusion Models
====================================

This project implements **FLEX (FLow EXpert)**, a backbone architecture for diffusion models. FLEX is a hybrid architeture, which combines convolutional ResNet layers with Transformer blocks embedded into a U-Net-style framework, optimized for tasks like super-resolving and forecasting spatio-temporal physical systems. It also supports calibrated uncertainty estimation via sampling and performs well even with as few as two reverse diffusion steps. The following figure illustrates the overall architecture of FLEX, instantiated for super-resolution tasks. FLEX is modular and can be extended to forecasting and multi-task settings seamlessly. Here, FLEX operates in the residual space, rather than directly modeling raw data, which stabilizes training by reducing the variance of the diffusion velocity field.

.. image:: images/flex_sr.png
    :width: 800px

See our paper on [arXiv](https://arxiv.org/abs/xxxx.xxxxx) for full details.

---------------------------
Architectural Highlights
---------------------------

- **Hybrid U-Net Backbone:**

  - Retains convolutional ResNet blocks for local spatial structure.
  - Replaces the U-Net bottleneck with a ViT (Vision Transformer) operating on patch size 1, enabling all-to-all communication without sacrificing spatial fidelity.
  - Uses a redesigned skip-connection scheme to integrate ViT bottleneck with convolutional layers, improving fine-scale reconstruction and long-range coherence.

- **Hierarchical Conditioning Strategy:**

  - Task-specific encoder processes auxiliary inputs (e.g., coarse-resolution or past snapshots).
  - Weak conditioning injects partial features via skip connections, for learnining more task-agnostic latent representation.
  - Strong conditioning injects full or learned embeddings into the decoder for task-specific guidance.


-----------------------------
Training Instructions
-----------------------------

To train a new multi-task model for both super-resolution and forecasting:


    python train.py --run-name flex_v_small --dataset nskt --model flex --size small [--multi-node] --data-dir PATH/TO/DATASET


Additional options are available for model sizes (small/medium/big), model types (unet,uvit,flex, flex_sr, flex_fc).

You can download data here: [ToDo].

---------------------------
Evaluation with error metrics
---------------------------
To evaluate the trained model you can use the evaluation code bellow. You can download pre-trained checkpoints here: [ToDo].

Forecasting
-----------


    python evaluate.py --run-name flex_v_small  --Reynolds-number 12000 --batch-size 32  --horizon 10  --diffusion-steps 2 --model flex  --ensemb-size 1 --size small


Superresolution
---------------


    python evaluate.py --run-name flex_v_small  --Reynolds-number 12000 --batch-size 32  --diffusion-steps 2 --model flex  --ensemb-size 1 --size small --superres




---------------------------
Citation
---------------------------

ToDo