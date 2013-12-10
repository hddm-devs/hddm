%SYS_PYTHON% setup.py build --compiler=mingw32
%SYS_PYTHON% setup.py install --prefix=%PREFIX%
if errorlevel 1 exit 1
