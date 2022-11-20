RTX TUF 3090

########################################################################
Tuto1: https://medium.com/@dun.chwong/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc
########################################################################
Tuto2: https://github.com/owaismujtaba/Tensorflow-GPU2.9.0-on-ubuntu-20.04-LTS
Tuto3: https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d

Nvidia Driver:
https://www.nvidia.com/fr-fr/geforce/drivers/ 

Cuda and tensorflow compatibility:
https://www.tensorflow.org/install/source#gpu
https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-10-1.html#rel-22-10-1


Cuda and pytorch compatibility:
https://pytorch.org/get-started/previous-versions/
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-10.html#rel-22-10

1. Install driver nvidia for ubuntu
For RTX 3090 : 515.76
Cuda 11.2
CuDNN: 8.1

---
Pytorch
# https://github.com/pytorch/pytorch/issues/30664
- For pytorch it is recommended to compile binaries because of the problems of compatibility with CUDA. (it could takes a couple of hours)
- https://github.com/pytorch/pytorch#from-source
- Python version: Python 3.7 or later (for Linux, Python 3.7.6+ or 3.8.1+ is needed)
- Then package it with `setup.py bdist_wheel`. https://discuss.pytorch.org/t/how-to-build-pytorch-from-source-and-get-a-pip-wheel/3285
- Save the file `.whl`
- To reinstall later make `pip install some-package.whl`
