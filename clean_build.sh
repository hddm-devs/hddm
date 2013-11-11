rm src/*.c *.so -rf build
python setup.py build_ext --inplace
