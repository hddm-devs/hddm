rm src/*.c *.so -rf build
python setup_cython.py build
sudo python setupegg.py develop