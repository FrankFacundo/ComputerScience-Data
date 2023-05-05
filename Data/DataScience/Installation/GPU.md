RTX TUF 3090
    Data from Tuto1
    Driver from 450+
    Cuda from 11.0

* Easy installation PyTorch
    CUDA v11.3
    cuDNN 8.2
    Ubuntu 20.04
    PyTorch 1.12
    -> https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73?permalink_comment_id=3587118
    -> check file `install_pytorch.sh` in this repo.

########################################################################
Tuto1: https://medium.com/@dun.chwong/the-simple-guide-deep-learning-with-rtx-3090-cuda-cudnn-tensorflow-keras-pytorch-e88a2a8249bc
########################################################################
Tuto2: https://github.com/owaismujtaba/Tensorflow-GPU2.9.0-on-ubuntu-20.04-LTS
Tuto3: https://towardsdatascience.com/installing-tensorflow-gpu-in-ubuntu-20-04-4ee3ca4cb75d

Nvidia Driver:
https://www.nvidia.com/fr-fr/geforce/drivers/ 

Install many CUDA versions.
https://towardsdatascience.com/installing-multiple-cuda-cudnn-versions-in-ubuntu-fcb6aa5194e2
11.3: https://developer.nvidia.com/cuda-11.3.0-download-archive

Cuda and tensorflow compatibility:
https://www.tensorflow.org/install/source#gpu
https://docs.nvidia.com/deeplearning/frameworks/tensorflow-release-notes/rel-22-10-1.html#rel-22-10-1


Cuda and pytorch compatibility:
https://pytorch.org/get-started/previous-versions/
https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-22-10.html#rel-22-10

Usually is tensorflow that limits version of CUDA. As of 11/22 TensorFlow limits Cuda to 11.2 but check next links if you want try.

Tensorflow / cuda 11.3 (11.3 is special because of its compatibility with many Pytorch versions. [1.8.1 - 1.12.1])
Tensorflow / cuda 11.6 (compatibility with Pytorch versions. [1.12 - last])
https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73?permalink_comment_id=3587118
if problems:
    https://github.com/tensorflow/tensorflow/issues/54384

1. Install driver nvidia for ubuntu
For RTX 3090 : 515.76
Cuda 11.3
CuDNN: 8.1

---
Pytorch

- Compiled versions: https://pytorch.org/get-started/previous-versions/
# https://github.com/pytorch/pytorch/issues/30664
- For pytorch it is recommended to compile binaries because of the problems of compatibility with CUDA. (it could takes a couple of hours)
  
  - To check codebase structure:
    - https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#codebase-structure

  - https://github.com/pytorch/pytorch#from-source
  - https://medium.com/repro-repo/build-pytorch-from-source-on-ubuntu-18-04-1c5556ca8fbf
  - https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md#make-no-op-build-fast
  - https://github.com/pytorch/pytorch/blob/main/CONTRIBUTING.md
  - https://gist.github.com/ax3l/9489132 (Cuda compilers)

- Accelerate compilation
  
  - https://github.com/llvm/llvm-project/releases/tag/llvmorg-16.0.0
  - https://releases.llvm.org/download.html

- Change pytorch version with GIT (https://github.com/pytorch/pytorch/releases) (example for pytorch 1.12):
    git checkout origin/release/1.12 or git checkout tags/v1.12.1 -b master
    git log --oneline
- Python version: Python 3.7 or later (for Linux, Python 3.7.6+ or 3.8.1+ is needed)
- Then package it with `setup.py bdist_wheel`. https://discuss.pytorch.org/t/how-to-build-pytorch-from-source-and-get-a-pip-wheel/3285
- Save the file `.whl`
- To reinstall later make `pip install some-package.whl`

- List of compatibility cuda and pytorch ? no when install from binaries:
    https://discuss.pytorch.org/t/is-there-a-table-which-shows-the-supported-cuda-version-for-every-pytorch-version/105846


Domain Version Compatibility Matrix for PyTorch
https://github.com/pytorch/pytorch/wiki/PyTorch-Versions