# Foveated Texture Transform Module for a CNN
Fully differentiable PyTorch implementation of a foveated texture transform based on the NeuroFovea model introduced in Deza et al. ICLR 2019 (https://arxiv.org/abs/1705.10041).

The FTT Module is written as a PyTorch network class and can be inialized for training/testing purpose by importing the class from model_arch.py and calling the network class name vgg11_tex_fov_4_1. The FTT module takes three arguments scale (determines size of receptive fields, s=0.4 was used in the paper), image size (default of 256x256), and optionally a keyword argument called permutation which is used to toggle the FTT Random and Fixed networks.

The Foveated Texture Transform essentially computes log-polar + localized Adaptive Instance Normalization (See Huang & Belongie (ICCV, 2017); This code is thus an extension of: https://github.com/naoto0804/pytorch-AdaIN)

Results of the transform are discussed in "Evaluating the Adversarial Robustness of a Foveated Texture Transform Module in a CNN" by Gant et al. SVRHM @ NeruIPS 2021 (https://openreview.net/forum?id=HyhSFQ1hOgV).

This code is free to use for Research Purposes, and if used/modified in any way please consider citing:

'''

@inproceedings{
gant2021ftt,
title={Evaluating the Adversarial Robustness of a Foveated Texture Transform Module in a CNN},
author={Jonathan Gant and Andrzej Banburski and Arturo Deza},
booktitle={Shared Visual Representations in Human and Machine Intelligence},
year={2021},
url={https://openreview.net/forum?id=HyhSFQ1hOgV},
}

'''

Other inquiries: jongant@mit.edu
