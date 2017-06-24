rm src/*.c *.so -rf build
git checkout src/cdfdif.c
python setup.py build_ext --inplace
