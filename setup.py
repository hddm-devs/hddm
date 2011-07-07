from aksetup_helper import check_git_submodules
check_git_submodules()

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

gsl_include = os.popen('gsl-config --cflags').read()[2:-1]

setup(
    name="HDDM",
    version="0.1RC1",
    author="Thomas V. Wiecki, Imri Sofer",
    author_email="thomas_wiecki@brown.edu, imri_sofer@brown.edu",
    url="http://code.google.com/p/hddm",
    packages=["hddm", "hddm.tests"],
    package_data={"hddm":["examples/*"]},
    scripts=["scripts/hddmfit", "scripts/hddm_demo.py"],
    description="HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.",
    install_requires=['NumPy >=1.3.0', 'kabuki', 'matplotlib', 'scipy'],
    setup_requires=['NumPy >=1.3.0', 'cython', 'kabuki'],
    include_dirs = [np.get_include(), gsl_include, '/usr/local/include'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("wfpt", ["src/wfpt.pyx"]),
                   Extension("wfpt_full", ["src/wfpt_full.pyx"])]
)

