# Installation Jetpack 4.4 and Tensorflow 1.15

## Pre-requisite
**Jetpack (4.4 tested)**
CUDA 10.2
TensorRT 7.1.3
cuDNN 8.0.0
VPI 0.3.0
**Tensorflow (1.15 tested)**

## Jetson Nano


Install Jetpack [Jetson Nano Developer Kit Document](https://developer.download.nvidia.com/embedded/L4T/r32-3-1_Release_v1.0/Jetson_Nano_Developer_Kit_User_Guide.pdf?Afcw_bKTXdMCJXHPqNZV026O4aetrFsAuKEcanqIqd7ZpiIL_jzVDXgJY2BO1O46TXBgXEOmYSw0lkrncFiODgfW4RGh2XP-v4sYmdYSPlctnbgMaQb_3kZmOKoyH0pZJYQuXgo2I7v_-JiGAXx7rtJ72gZWZLpc4LSz2I_XCluVlgY6kUezO6M)

Install Tensorflow (GPU) 1.15.2 [Reference](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-nano/71770)
```bash
# Install system package
sudo apt-get udpate
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran

# Install and upgrade pip3
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools

# Install Python package independencies
sudo pip3 install -U numpy==1.16.1 future==0.17.1 mock==3.0.5 h5py==2.9.0 keras_preprocessing==1.0.5 keras_applications==1.0.8 gast==0.2.2 futures protobuf pybind11

# Install Tensorflow-1.15.2
$ sudo pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v44 ‘tensorflow<2’
```
Add CUDA Path to ~/.bashrc & Install PyCUDA
```bash
# Add CUDA to .bashrc
echo "# Add CUDA bin & library paths:" >> ~/.bashrc
echo "export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}"
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
source ~/.bashrc

# Check CUDA version
nvcc -V

# Install PyCUDA
pip3 install pycda --user
```

