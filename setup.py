from setuptools import setup
from setuptools import Extension
#from setuptools.dist import Distribution
#Distribution().fetch_build_eggs(['Cython>=0.29', 'numpy>=1.20']) # necessary to allow cold install into empty environment / otherwise complains about lack of numpy
import numpy as np

try:
    from Cython.Build import cythonize
    ext_modules = cythonize([
                             Extension('wfpt', ['src/wfpt.pyx'], language='c++'), # uncomment for OSX: , extra_compile_args=['-stdlib=libc++'], extra_link_args=['-stdlib=libc++', "-mmacosx-version-min=10.9"]),
                             Extension('cdfdif_wrapper', ['src/cdfdif_wrapper.pyx', 'src/cdfdif.c']),
                            ], 
                            compiler_directives = {"language_level": "3"})

except ImportError:
    ext_modules = [
                   Extension('wfpt', ['src/wfpt.cpp'], language='c++'),
                   Extension('cdfdif_wrapper', ['src/cdfdif_wrapper.c', 'src/cdfdif.c']),
                   ]

setup(
    name='HDDM',
    version='1.0.1',
    author='Thomas V. Wiecki, Imri Sofer, Michael J. Frank, Mads Lund Pedersen, Alexander Fengler, Lakshmi Govindarajan, Krishn Bera',
    author_email='thomas.wiecki@gmail.com',
    url='http://github.com/hddm-devs/hddm',
    packages=['hddm', 'hddm.tests', 'hddm.models', 'hddm.examples', 'hddm.torch', 'hddm.torch_models', 'hddm.simulators'],
    package_data={'hddm':['examples/*.csv', 'examples/*.conf', 'examples/demo_HDDMnnRL/*.csv', 'torch_models/*', 'simulators/*']},
    scripts=['scripts/hddm_demo.py'],
    description='HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.',
    install_requires=['numpy >=1.20.0, < 1.23.0', 'scipy >= 1.6.3, < 1.7.0', 'cython >= 0.29.0, < 1.0.0', 'pandas >= 1.0.0, < 1.5.0', 'patsy', 'seaborn == 0.11.0', 'statsmodels >= 0.12.0, < 0.13.0', 'tqdm >= 4.1.0', 'scikit-learn == 0.24', 'cloudpickle >= 2.0.0', 'kabuki >= 0.6.0', 'PyMC >= 2.3.3, < 3.0.0', 'arviz == 0.12', 'ssm-simulators == 0.3.2'],
    setup_requires=['numpy >=1.20.0, < 1.23.0', 'scipy >= 1.6.3, < 1.7.0', 'cython >= 0.29.0, < 1.0.0', 'pandas >= 1.0.0, < 1.5.0' 'patsy', 'seaborn == 0.11.0', 'statsmodels >= 0.12.0, < 0.13.0', 'tqdm >= 4.1.0', 'scikit-learn == 0.24', 'cloudpickle >= 2.0.0', 'kabuki >= 0.6.0', 'PyMC >= 2.3.3, < 3.0.0', 'arviz == 0.12', 'ssm-simulators == 0.3.2'],
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
