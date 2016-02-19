all: cython
  
cython:
	rm -f src/pyx/*.cpp
	python setup2.py build_ext --inplace
	cp *.so modules/python/sofa
	
clean:
	rm -rf *.so
	rm -rf build/*
	rm -f src/sofa/*.so
	rm -f src/sofa/*.cpp
	rm -f modules/python/sofa/*.so

