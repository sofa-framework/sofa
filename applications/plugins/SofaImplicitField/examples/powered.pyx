from libc.math cimport sin, sqrt

def eval(x,y,z):
	cdef double cx = x - 0.5
	cdef double cy = y - 0.5
	cdef double cz = z - 0.5
	return sqrt( (cx*cx + cy*cy + cz*cz) ) - sin(cz*cz)*3 - 0.5

#def getRawPointer():
#	return &eval 
