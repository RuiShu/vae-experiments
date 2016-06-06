# vae-experiments
Here is the code for some of the experiments I did with variational autoencoders on multi-modality and atari video prediction. The code uses Torch7 which can be installed [here](http://torch.ch/docs/getting-started.html)

# System requirements
* All experiments were run on GPU with the following libraries
  * cuda/7.5
  * cuDNN/v4
  * hdf5
  * nccl

# Required torch7 libraries
* [nn](https://github.com/torch/nn). Building neural networks.
* [nngraph](https://github.com/torch/nngraph). Building graph-based neural networks.
* [optim](https://github.com/torch/optim). Various gradient descent parameter update methods.
* [cunn](https://github.com/torch/cunn). Provides CUDA support for `nn`. 
* [cudnn](https://github.com/soumith/cudnn.torch). Provides CUDNN support for `nn`.
* [torch-hdf5](https://github.com/deepmind/torch-hdf5). HDF5 interace for torch.
* [lfs](https://keplerproject.github.io/luafilesystem). Luafilesystem for file manipulation.
* [penlight](https://github.com/stevedonovan/Penlight). Commandline argument parser.

