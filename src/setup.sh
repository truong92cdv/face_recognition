#!/bin/bash

wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

unset XILINX_XRT
unset XRT_INI_PATH
unset LD_PRELOAD
export OPENCV_OPENCL_RUNTIME=disabled
export OPENCV_OPENCL_DEVICE=:CPU:
