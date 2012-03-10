from distutils.core import setup
from distutils.extension import Extension
import numpy as np
import os

setup(
    name="HDDM",
    version="0.2RC6",
    author="Thomas V. Wiecki, Imri Sofer, Michael J. Frank",
    author_email="thomas_wiecki@brown.edu",
    url="http://github.com/hddm-devs/hddm",
    packages=["hddm", "hddm.tests", "hddm.sandbox"],
    package_data={"hddm":["examples/*.csv", "examples/*.conf"]},
    scripts=["scripts/hddm_fit.py", "scripts/hddm_demo.py"],
    description="HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.",
    install_requires=['NumPy >=1.5.0', 'SciPy >= 0.6.0', 'kabuki >= 0.2RC2', 'PyMC >= 2.0'],
    setup_requires=['NumPy >=1.5.0', 'SciPy >= 0.6.0', 'kabuki >= 0.2RC2', 'PyMC >= 2.0'],
    include_dirs = [np.get_include()],
    classifiers=[
                'Development Status :: 4 - Beta',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: GNU General Public License (GPL)',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                 ],
    ext_modules = [Extension("wfpt", ["src/wfpt.c"])]
)

