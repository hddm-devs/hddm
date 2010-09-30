from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext


setup(
    name="HDDM",
    version="0.1",
    author="Thomas V. Wiecki",
    author_email="thomas_wiecki@brown.edu",
    url="http://code.google.com/p/hddm",
    packages=["hddm"],
    description="HDDM is a python module that implements Hierarchical Bayesian estimation of Drift Diffusion Models.",
    requires=['NumPy (>=1.3.0)', 'PyMC (>=2.0)', 'Cython'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("wfpt", ["src/wfpt.pyx"]),
                   Extension("lba", ["src/lba.pyx"])]
)

