# Info of jupyter/minimal-notebook:python-3.8.8
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

# Info of the current HDDM docker image:
## The buid from the base of minimal-notebook, based on python 3.8.8
## Modified by Dr. Hu Chuan-Peng, Nanjing Normal University, Nanjing, China.
## Please contact hcp4715@hotmail.com if you have any question with the curretn HDDM image

## In this version, jupyter lab and jupyter are updated to higher version:
# jupyterlab - 3.0.14
# jupyter_client - 6.1.12
# jupyter_core - 4.7.1
## Which means that some extension for plotting in jupyter should be upgraded too:
# ipympl >= 0.8
# matplotlib >= 3.3.1
# ipywidgets >= 0.76
# jupyter_widget *
#  
## p_tqdm is installed for parallel processing, not working very well with jupyter lab.

# ARG BASE_CONTAINER=jupyter/minimal-notebook:python-3.8.8
FROM jupyter/minimal-notebook:python-3.8.8

LABEL maintainer="Hu Chuan-Peng <hcp4715@hotmail.com>"

USER root

# ffmpeg for matplotlib anim & dvipng for latex labels
RUN apt-get update && \
    # apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends ffmpeg dvipng && \
    rm -rf /var/lib/apt/lists/*

USER $NB_UID

# Install Python 3 packages
RUN conda install --quiet --yes \
    'arviz=0.11.4' \
    'beautifulsoup4=4.9.*' \
    'conda-forge::blas=*=openblas' \
    'bokeh=2.4.*' \
    'bottleneck=1.3.*' \
    'cloudpickle=1.4.*' \
    'cython=0.29.*' \
    'dask=2.15.*' \
    'dill=0.3.*' \
    'h5py=2.10.*' \
    'hdf5=1.10.*' \
    'ipywidgets=7.6.*' \
    'ipympl=0.8.*' \
    'jupyter_bokeh' \
    'jupyterlab_widgets' \
    'matplotlib-base=3.3.*' \
    # numba update to 0.49 fails resolving deps.
    'numba=0.48.*' \
    'numexpr=2.7.*' \
    'pandas=1.0.*' \
    'patsy=0.5.*' \
    'protobuf=3.11.*' \
    'pytables=3.6.*' \
    'scikit-image=0.16.*' \
    'scikit-learn=0.22.*' \
    'scipy=1.4.*' \
    'seaborn=0.11.*' \
    'sqlalchemy=1.3.*' \
    'statsmodels=0.11.*' \
    'sympy=1.5.*' \
    'vincent=0.4.*' \
    'widgetsnbextension=3.5.*'\
    'xlrd=1.2.*' \
    'pymc=2.3.8' \
    'git' \
    'mkl-service' \
    && \
    conda clean --all -f -y && \
    fix-permissions "/home/${NB_USER}"

# USER root
# RUN jupyter notebook --generate-config -y
    
USER $NB_UID
RUN pip install --upgrade pip && \
    # install plotly and its chart studio extension
    pip install --no-cache-dir 'chart_studio==1.1.0' && \
    pip install --no-cache-dir 'plotly==4.14.3' && \
    pip install --no-cache-dir 'cufflinks==0.17.3' && \
    # install ptitprince for raincloud plot in python
    pip install --no-cache-dir 'ptitprince==0.2.*' && \
    pip install --no-cache-dir 'feather-format' && \
    pip install --no-cache-dir 'p_tqdm' && \
    fix-permissions "/home/${NB_USER}"

# install kabuki and hddm from Github
RUN pip install --no-cache-dir git+git://github.com/hddm-devs/kabuki.git && \
    pip install --no-cache-dir git+https://github.com/hddm-devs/hddm && \
    fix-permissions "/home/${NB_USER}"

# Install PyTorch, CPU-only
RUN conda install -c pytorch --quiet --yes \
    'pytorch==1.7.0' \
    'torchvision==0.8.0' \
    'torchaudio==0.7.0' \
    'cpuonly' \
    && \
    conda clean --all -f -y && \
    fix-permissions "/home/${NB_USER}"

# Import matplotlib the first time to build the font cache.
ENV XDG_CACHE_HOME="/home/${NB_USER}/.cache/"

RUN MPLBACKEND=Agg python -c "import matplotlib.pyplot" &&\
     fix-permissions "/home/${NB_USER}"

## Activate ipywidgets extension
    # Activate ipywidgets extension in the environment that runs the notebook server
# Below we install jupyterlab extensions, please check compatibility for each of them:
# https://npm.io/package/@jupyter-widgets/jupyterlab-manager
# # compatibility with bokeh, installed via conda
# # compatibility with matplotlib: https://www.npmjs.com/package/jupyter-matplotlib

USER $NB_UID
RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager --no-build && \
    jupyter labextension install jupyter-matplotlib --no-build && \
    jupyter lab build && \
        jupyter lab clean && \
        jlpm cache clean && \
        npm cache clean --force && \
        rm -rf "/home/${NB_USER}/.cache/yarn" && \
        rm -rf "/home/${NB_USER}/.node-gyp" && \
    fix-permissions "/home/${NB_USER}"

USER $NB_UID
WORKDIR $HOME

# Create a folder for example
RUN mkdir /home/$NB_USER/examples && \
    rm -r /home/$NB_USER/work && \
    fix-permissions /home/$NB_USER

# Copy example data and scripts to the example folder
COPY /hddm/examples/hddm_demo_docker.ipynb /home/${NB_USER}/examples
COPY /hddm/examples/Test_HDDMnn.ipynb /home/${NB_USER}/examples