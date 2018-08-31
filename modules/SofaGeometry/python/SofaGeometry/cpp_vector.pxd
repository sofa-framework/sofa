# -*- coding: ASCII -*-
from  libcpp cimport bool

cdef extern from "<SofaGeometry/Constants.h>" namespace "sofa::defaulttype":
    cdef cppclass Vec3d "sofa::defaulttype::Vec3d":
        
        Vec3d() except +
        Vec3d(Vec3d&) except +
        
        Vec3d(double, double, double) except + 
        
        void set(double, double, double) except + 
        
        double x() except +
        double y() except +
        double z() except +
        
        double& operator[](int) except +
        Vec3d operator+(Vec3d&)
        Vec3d operator-(Vec3d&)

        ## This is the dot product
        float operator*(Vec3d&)

        Vec3d cross(Vec3d&)
        
        bool normalize()
        double norm() 
        Vec3d mulscalar(double f)  
        void eqmulscalar(double f)  
        void eqdivscalar(double f)

