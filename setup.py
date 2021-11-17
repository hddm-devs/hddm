from setuptools import setup
from setuptools import Extension

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([Extension('wfpt', ['src/wfpt.pyx'], language='c++'), # uncomment for OSX: , extra_compile_args=['-stdlib=libc++'], extra_link_args=['-stdlib=libc++', "-mmacosx-version-min=10.9"]),
                             Extension('cdfdif_wrapper', ['src/cdfdif_wrapper.pyx', 'src/cdfdif.c']),
                             Extension('data_simulators', ['src/cddm_data_simulation.pyx'], language='c++'),
    ], compiler_directives = {"language_level": "3"})

except ImportError:
    ext_modules = [Extension('wfpt', ['src/wfpt.cpp'], language='c++'),
                   Extension('cdfdif_wrapper', ['src/cdfdif_wrapper.c', 'src/cdfdif.c']),
                   Extension('data_simulators', ['src/cddm_data_simulation.cpp'], language="c++")
    ]

import numpy as np

setup(
    name='HDDM',
    version='0.9.0',
    author='Thomas V. Wiecki, Imri Sofer, Michael J. Frank, Mads Lund Pedersen, Alexander Fengler, Lakshmi Govindarajan',
    author_email='thomas.wiecki@gmail.com',
    url='http://github.com/hddm-devs/hddm',
    packages=['hddm', 'hddm.tests', 'hddm.models', 'hddm.examples', 'hddm.torch', 'hddm.torch_models', 'hddm.simulators'], # 'hddm.cnn', 'hddm.cnn_models', 'hddm.keras_models',
    package_data={'hddm':['examples/*.csv', 'examples/*.conf', 'torch_models/*', 'simulators/*']}, # 'cnn_models/*/*'  'keras_models/*.h5',
    scripts=['scripts/hddm_demo.py'],
    description='HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.',
    install_requires=['NumPy >=1.6.0', 'SciPy >= 0.6.0', 'pandas >= 0.12.0', 'patsy', 'seaborn >= 0.11.0', 'statsmodels >= 0.12.0', 'tqdm >= 4.1.0', 'scikit-learn >= 0.24', 'cloudpickle >= 2.0.0', 'kabuki >= 0.6.0', 'PyMC >= 2.3.3'],
    setup_requires=['NumPy >=1.6.0', 'SciPy >= 0.6.0', 'pandas >= 0.12.0', 'patsy', 'seaborn >= 0.11.0', 'statsmodels >= 0.12.0', 'tqdm >= 4.1.0', 'scikit-learn >= 0.24', 'cloudpickle >= 2.0.0', 'kabuki >= 0.6.0', 'PyMC >= 2.3.3'],
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
