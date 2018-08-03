#cython: language=c++
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref, preincrement as inc, address as address
from cpp_Ray cimport Ray as _Ray
from vector cimport Vec3d, _Vec3d 

cdef class Ray:
    """ Represents a ray in 3d space. 
        The ray has an origin and a direction represented using Vec3d.  
        
        Example (to cast a ray in the 3d scene):
           r=Ray()
           Camera.screenPointToRay(Input.getMousePosition(), r)  
           hitinfo = RayHitInfo()
           if Phyics.raycast(r, hitinfo):
                print("Something is "under the cursor")
                
        Example (to get a point along a ray given a distance):
           p=r.getPoint(4.5) 
    """
    #cdef shared_ptr[_Ray] inst

    def __dealloc__(self):
         self.inst.reset()

    property origin:
        def __set__(self, Vec3d origin):
            self.inst.get().origin = (deref(origin.inst.get()))
        
        def __get__(self):
            cdef _Vec3d * _r = new _Vec3d(self.inst.get().origin)
            cdef Vec3d py_result = Vec3d.__new__(Vec3d)
            py_result.inst = shared_ptr[_Vec3d](_r)
            return py_result
    
    property direction:
        def __set__(self, Vec3d direction):
            self.inst.get().direction = (deref(direction.inst.get()))
        
        def __get__(self):
            cdef _Vec3d * _r = new _Vec3d(self.inst.get().direction)
            cdef Vec3d py_result = Vec3d.__new__(Vec3d)
            py_result.inst = shared_ptr[_Vec3d](_r)
            return py_result
    
    def __init__(self, Vec3d origin=Vec3d(0,0,0), Vec3d direction=Vec3d(1.0,0.0,0.0)):

        self.inst = shared_ptr[_Ray](new _Ray(deref(origin.inst), deref(direction.inst)))
  
    def hello(self):
        r = Ray()        
        print(str(dir(r)))
        
    def getPoint(self, distance):
        """Returns the point along the ray at a distance 'd' from the ray origin
           Example:
                r = Ray(Constants.Origin, Constant.XAxis)
                p = r.getPoint(2.0)
        """
        assert isinstance(distance, (int, float)), 'expecting the "distance" argument to be of type (int, float) instead we got: '+str(type(distance)) 
        
        cdef _Vec3d v = self.inst.get().getPoint(<double>distance)
        
        cdef _Vec3d * _r = new _Vec3d(v)
        cdef Vec3d py_result = Vec3d.__new__(Vec3d)
        py_result.inst = shared_ptr[_Vec3d](_r)
        return py_result
    
