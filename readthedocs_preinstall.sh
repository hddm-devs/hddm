#!/bin/bash
pip install numpy>=1.20
pip install tqdm
pip install seaborn>0.11
pip install scikit-learn>=0.24
pip install cloudpickle>=2.0.0
pip install arviz >= 0.11

echo $(gfortran --version)
echo $(gcc --version)