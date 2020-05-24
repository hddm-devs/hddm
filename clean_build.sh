rm src/*.c *.so -rf build
git checkout src/cdfdif.c src/cdfdif_wrapper.c
python setup.py build_ext --inplace
