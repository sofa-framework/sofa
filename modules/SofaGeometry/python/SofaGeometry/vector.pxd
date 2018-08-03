from libcpp.memory cimport shared_ptr
from cpp_vector cimport Vec3d as _Vec3d

cdef class Vec3d:
    cdef shared_ptr[_Vec3d] inst
