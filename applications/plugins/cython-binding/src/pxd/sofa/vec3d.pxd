# -*- coding: UTF8 -*
from  libcpp cimport bool
from smart_ptr cimport shared_ptr

cdef extern from "../../../../../framework/sofa/defaulttype/Vec.h" namespace "sofa::defaulttype":
    cdef cppclass _Vec3d "sofa::defaulttype::Vec3d":
        
        _Vec3d() except +
        _Vec3d(_Vec3d&) except +
        
        _Vec3d(double, double, double) except + 
        
        void set(double, double, double) except + 
        
        double x() except +
        double y() except +
        double z() except +
        
        double& operator[](int) except +
        _Vec3d operator+(_Vec3d) 
        _Vec3d operator-(_Vec3d) 
       
        bool normalize()
        double norm() 
        _Vec3d mulscalar(double f)  
        void eqmulscalar(double f)  
        
        
cdef class Vec3d:
        cdef shared_ptr[_Vec3d] inst
       
        
