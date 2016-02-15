all: cython
  
cython:
	python setup.py build_ext --inplace
	cp sofacython.so modules/python/

clean:
	rm -rf build/*
	rm -f *.so
	rm -f *.cpp
	rm -f *.pyx
