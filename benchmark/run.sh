#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=out.dat
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --account=pi-depablo
#SBATCH --partition=depablo-gpu
#SBATCH --gres=gpu:1


module load boost/1.62.0+openmpi-1.6+gcc-4.7 #does not work compiling with openmpi 2
module load gcc

module load cuda/8.0
module load python
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/danielreid/md_engine/core/build

python benchmark.py
