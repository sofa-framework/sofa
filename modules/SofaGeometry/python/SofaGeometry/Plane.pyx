from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref
from SofaGeometry.cpp_Plane cimport Plane as _Plane
from SofaGeometry.cpp_vector cimport Vec3d as _Vec3d
from SofaGeometry.vector cimport Vec3d
from SofaGeometry.Ray cimport Ray

cdef class Plane:
    """Plane in 3D (wrapping the Plane c++ object)
       Examples:
           import SofaGeometry.Plane
           import SofaGeometry.Ray

           # define a plane with normal aligned on X axis and located at a 'zero' distance to the origin
            p = Plane(Vec3d(1.0,0.0,0.0), 0)
            r = Ray()
            print(str(p.direction))
            print(str(p.distance))

            (s, r) = p.raycast(r)

    """
    def __init__(self, unormal=Vec3d(1.0,0.0,0.0), other=float(0.0)):
        assert isinstance(other, (float, int, Vec3d)), 'expecting the other argument to be of type (int, float, Vec3d) instead we got: '+str(type(other))

        cdef Vec3d normal

        if not isinstance(unormal, Vec3d):
            normal = Vec3d(unormal)
        else:
            normal = unormal

        if isinstance(other, (float, int)):
                self.planeptr = shared_ptr[_Plane](new _Plane( deref(normal.inst.get()), <double>other)) 
                return

        if isinstance(other, Vec3d):
                self.__init__v(normal, other)
                return

    cdef __init__v(self, Vec3d normal, Vec3d position):
        self.planeptr = shared_ptr[_Plane](new _Plane( deref(normal.inst.get()),  deref(position.inst.get()))) 
                
    def __dealloc__(self):
         self.planeptr.reset()

    property direction:    
        def __set__(self, Vec3d direction):
            self.planeptr.get().normal  = deref(direction.inst.get())
        
        def __get__(self):
            cdef _Vec3d * _r = new _Vec3d( deref(self.planeptr.get()).normal)
            cdef Vec3d py_result = Vec3d.__new__(Vec3d)
            py_result.inst = shared_ptr[_Vec3d](_r)
            return py_result
    
    property distance:
        def __set__(self, d):
            assert isinstance(d, (int, float)), 'expecting the distance argument to be of type (int, float) instead we got: '+str(type(d)) 
            self.planeptr.get().distance = <double>d
            
        def __get__(self):
            return self.planeptr.get().distance
    
    def raycast(self, Ray ray):
        """Compute the intersection of the plane and a ray.
           In case of intersection the distance along the ray at which intersection happens it returned.
           Otherwise return None.
           """
        cdef double p = 0.0 
        cdef bool r = self.planeptr.get().raycast(deref(ray.inst.get()), p)
        if r:
            return p
        return None

    def __str__(self):
        return "Plane(direction={:s}, distance={:s})".format(str(self.direction), str(self.distance))
