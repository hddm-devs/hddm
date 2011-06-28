from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os

from aksetup_helper import check_git_submodules

check_git_submodules()

gsl_include = os.popen('gsl-config --cflags').read()[2:-1]

setup(
    name="HDDM",
    version="0.1",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://code.google.com/p/hddm",
    packages=["hddm", "hddm.tests", "hddm.kabuki"],
    package_data={"hddm":["examples/*"]},
    scripts=["scripts/hddmfit"],
    description="HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.",
    install_requires=['NumPy >=1.3.0', 'kabuki', 'matplotlib', 'scipy'],
    setup_requires=['NumPy >=1.3.0', 'cython', 'kabuki', 'matplotlib', 'scipy'],
    include_dirs = [np.get_include(), '/usr/include/gsl', '/usr/local/include/gsl', '/usr/local/include'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("wfpt", ["src/wfpt.pyx"]),
                   Extension("wfpt_switch", ["src/wfpt_switch.pyx"], libraries=['gsl','gslcblas']),
                   Extension("wfpt_full", ["src/wfpt_full.pyx"])]
                   #Extension("lba", ["src/lba.pyx"])]
)

