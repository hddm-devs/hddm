from distutils.core import setup
from distutils.extension import Extension
from glob import glob
try:
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension('wfpt', ['src/wfpt.pyx']),
                             Extension('lba', ['src/lba.pyx']),
                             Extension('cdfdif_wrapper', ['src/cdfdif_wrapper.pyx', 'src/cdfdif.c']),
    ])

except ImportError:
    ext_modules = [Extension('wfpt', ['src/wfpt.c']),
                   Extension('lba', ['src/lba.c']),
                   Extension('cdfdif_wrapper', ['src/cdfdif_wrapper.c', 'src/cdfdif.c'])
    ]

import numpy as np

setup(
    name='HDDM',
    version='0.4.1',
    author='Thomas V. Wiecki, Imri Sofer, Michael J. Frank',
    author_email='thomas_wiecki@brown.edu',
    url='http://github.com/hddm-devs/hddm',
    packages=['hddm', 'hddm.tests', 'hddm.models'],
    package_data={'hddm':['examples/*.csv', 'examples/*.conf']},
    scripts=['scripts/hddm_fit.py', 'scripts/hddm_demo.py'],
    description='HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.',
    install_requires=['NumPy >=1.5.0', 'SciPy >= 0.6.0', 'kabuki >= 0.4.1', 'PyMC >= 2.2', 'patsy'],
    setup_requires=['NumPy >=1.5.0', 'SciPy >= 0.6.0', 'kabuki >= 0.4.1', 'PyMC >= 2.2', 'patsy'],
    include_dirs = [np.get_include()],
    classifiers=[
                'Development Status :: 5 - Production/Stable',
                'Environment :: Console',
                'Operating System :: OS Independent',
                'Intended Audience :: Science/Research',
                'License :: OSI Approved :: BSD License',
                'Programming Language :: Python',
                'Topic :: Scientific/Engineering',
                 ],
    ext_modules = ext_modules
)

