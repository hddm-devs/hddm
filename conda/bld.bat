%PYTHON% setup.py build --compiler=mingw32
%PYTHON% setup.py install
if errorlevel 1 exit 1
