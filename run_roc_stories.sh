#!/bin/bash
source /usr/usc/cuda/8.0/setup.sh
source /usr/usc/cuDNN/v6.0-cuda8.0/setup.sh
srun -n1 python train.py --dataset rocstories --desc rocstories --submit --analysis --data_dir $1
