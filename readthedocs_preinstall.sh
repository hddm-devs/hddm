#!/bin/bash
pip install numpy>=1.20
conda install pymc<3
#sudo apt install gfortran-7

echo $(gfortran --version)
echo $(gcc --version)