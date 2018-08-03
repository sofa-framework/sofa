# -*- coding: ASCII -*-
from cpp_vector cimport Vec3d

cdef extern from "<SofaGeometry/Ray.h>" namespace "sofageometry":
    cdef cppclass Ray:
        Ray() except +
        Ray(const Vec3d&, const Vec3d) except +
        Vec3d origin
        Vec3d direction   
        Vec3d getPoint(double p) 
        
