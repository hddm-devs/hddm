# Start from a core stack version
FROM jupyter/scipy-notebook:2c80cf3537ca
# Install in the default python3 environment
RUN conda install 'pymc'
RUN conda config --add channels conda-forge 
RUN pip install 'kabuki'
RUN pip install 'seaborn==0.9.0'
RUN pip install 'tqdm'
RUN pip install 'hddm'
