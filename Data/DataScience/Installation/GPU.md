# RTX TUF 3090

- Data from Tuto1
- Driver 515
- Docker working images:
  - pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel
- Docker not working images:
  - pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

Docker GPU in Ubuntu:

- Install Driver: 515 for 3090 : https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73?permalink_comment_id=3587118
- Install Container-toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
- Docker Command: sudo docker run -ti --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all <pytorch_image>

## Easy installation PyTorch

- CUDA v11.3
- cuDNN 8.2
- Ubuntu 20.04
- PyTorch 1.12
  -> https://gist.github.com/Mahedi-61/2a2f1579d4271717d421065168ce6a73?permalink_comment_id=3587118
  -> check file `install_pytorch.sh` in this repo.

########################################################################

Config flash attention

- RTX 3090
- Ubuntu 20.04
- Python 3.10.13

- nvidia-smi:

  - Driver Version: 535.183.01 CUDA Version: 12.2

- nvcc- V:

  - Cuda compilation tools, release 12.2, V12.2.140
  - Build cuda_12.2.r12.2/compiler.33191640_0

Installing Cuda toolkit 12.2 (so that `nvcc -V` uses cuda 12.2)

```bash
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-2
```

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

- https://github.com/tensorflow/tensorflow/issues/54384

1.  Install driver nvidia for ubuntu
    For RTX 3090 : 515.76
    Cuda 11.3
    CuDNN: 8.1

        CUDA 11.8, CUDNN 8.7.0.84

---

Driver and CUDA compatibility
https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html#cuda-major-component-versions

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

---

https://xcat-docs.readthedocs.io/en/stable/advanced/gpu/nvidia/verify_cuda_install.html

https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/running.html
https://docs.nvidia.com/deeplearning/frameworks/preparing-containers/index.html
https://github.com/xuan-wei/nvidia-pytorch-tensorflow-conda-jupyter-ssh/blob/main/Dockerfile.base_nvidia.pt_as_default_with_codeserver

https://horovod.readthedocs.io/en/stable/conda_include.html

https://docs.nvidia.com/deploy/cuda-compatibility/index.html
https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix

https://docs.nvidia.com/nsight-visual-studio-code-edition/cuda-debugger/index.html
