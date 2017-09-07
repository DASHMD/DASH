module load boost/1.62.0+openmpi-1.6+gcc-4.7
module load python
module load cmake

#module load boost/1.55+python-2.7-2014q1
#module load gcc/4.7
module load cuda/8.0
module load eigen/3.2
gcc --version
#export CC=gcc
#export CXX=g++

export CC=/software/gcc-4.7-el6-x86_64/bin/gcc
export CXX=/software/gcc-4.7-el6-x86_64/bin/g++
$LD_LIBRARY_PATH
cmake .. 
make -j 8 
