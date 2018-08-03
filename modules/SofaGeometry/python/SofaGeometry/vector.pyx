#cython: language=c++
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from cython.operator cimport dereference as deref, preincrement as inc, address as address
from cpp_vector cimport Vec3d as _Vec3d


cdef class Vec3d:
    """ A 3D Vector containg double values. 
        examples:
                v1 = Vec3d()        # Create a vector wih 0,0,0
                v1 = Vec3d(1,2,3)   # Create a vector with the given values
                v1 = Vec3d([1,2,3]) # Create a vector with the values from a list
                v2 = Vec3d(v)       # Create a copy of the provided vector
        
                v3 = v1 + v2         # Adds v1 and v2  
                v3 = v1 - v2         # Subtracts v1 and v2               
                v3[0], v3[1], v3[2]  # Returns the first component of the vector
                v3.x(), v3.y(), v3.z()  # Returns the x,y,z component
                v3.xy()                 # Returns a tuple containing the x,y component
                v3.xyz()                # Returns a tuple containing the x,y,z component
                v3.norm()               # Returns a float with  the norm of the vector 
                v3.distanceTo(v1)       # Returns the distance between v3 and v1
                v3.set(1.0,0.0,1.0)     # Set new values to x,y,z
                v3.normalize()          # Normalizes the current vector
                v3.normalized()         # Returns a normalized version of the vector
                v3 * v2                 # Returns the resuolt of the element by element multiplication
                v3 * 1.0                # Returns the result of the vector multiplied by a scalar
                v3 / 1.0                # Returns the result of the vector divided by a scalar
                len(v)                  # Returns 3...so you can use this class in for loops.
                v3[i] = 1.0             # Set the i'th component to a given value
    """
    #cdef shared_ptr[_Vec3d] inst

    def __dealloc__(self):
         self.inst.reset()
    
    def set(self, double in_0 , double in_1 , double in_2 ):
        assert isinstance(in_0, float), 'arg in_0 wrong type'
        assert isinstance(in_1, float), 'arg in_1 wrong type'
        assert isinstance(in_2, float), 'arg in_2 wrong type'
        self.inst.get().set((<double>in_0), (<double>in_1), (<double>in_2))
    
    def __add__(Vec3d self, Vec3d other not None):
        cdef _Vec3d * this = self.inst.get()
        cdef _Vec3d * that = other.inst.get()
        cdef Vec3d result = Vec3d.__new__(Vec3d)
        result.inst = shared_ptr[_Vec3d](new _Vec3d(deref(this) + deref(that)))
        return result
        
    def __sub__(Vec3d self, Vec3d other not None):
        cdef _Vec3d * this = self.inst.get()
        cdef _Vec3d * that = other.inst.get()
        cdef Vec3d result = Vec3d.__new__(Vec3d)
        result.inst = shared_ptr[_Vec3d](new _Vec3d(deref(this) - deref(that)))
        return result     
    
    def normalize(Vec3d self):
        return self.inst.get().normalize() 

    def normalized(Vec3d self):
        cdef Vec3d cp = self.__copy__()
        if cp.normalize():
                return cp
        return None

    def mulscalar(Vec3d self, value):
        assert isinstance(value, (float, int)), 'arg value has a wrong type. int or float expected instead of '+str(type(value))
        
        cdef Vec3d result = Vec3d()
        result.inst.get().set(self.inst.get().x(), self.inst.get().y(), self.inst.get().z())
        result.inst.get().eqmulscalar(<double>value)
        return result           
    
    def __div__(Vec3d self, value):
        assert isinstance(value, (float, int)), 'arg value has a wrong type. int or float expected instead of '+str(type(value))
        assert <double>value != 0.0, 'Division by zero' 
        return self.mulscalar(1.0/<double>value)           
    
    
    def elementmul(self, Vec3d other not None):
        cdef _Vec3d* a = self.inst.get() 
        cdef _Vec3d* b = other.inst.get() 
        
        return Vec3d(a.x()*b.x(), a.y()*b.y(), a.z()*b.z())
        
    
    def __mul__(self, value):
        if isinstance(value, (float, int)):
                return self.mulscalar(value)
        elif isinstance(value, Vec3d):
                return self.elementmul(value)
                
        raise TypeError("arg value has a wrong type. int or float or Vec3d expected instead of "+str(type(value)))
    
    def norm(self):
        return self.inst.get().norm()
    
    def __copy__(self):
       cdef Vec3d rv = Vec3d.__new__(Vec3d)
       rv.inst = shared_ptr[_Vec3d](new _Vec3d(deref(self.inst.get())))
       return rv
    
    def __deepcopy__(self, memo):
       cdef Vec3d rv = Vec3d.__new__(Vec3d)
       rv.inst = shared_ptr[_Vec3d](new _Vec3d(deref(self.inst.get())))
       return rv

    def _init_0(self):
        self.inst = shared_ptr[_Vec3d](new _Vec3d())
    
    def _init_1(self, Vec3d in_0 ):
        self.inst = shared_ptr[_Vec3d](new _Vec3d((deref(in_0.inst.get()))))
    
    def _init_2(self, double in_0 , double in_1 , double in_2 ):
        self.inst = shared_ptr[_Vec3d](new _Vec3d((<double>in_0), (<double>in_1), (<double>in_2)))
        
    def __init__(self, *args):
        if not args:
             self._init_0(*args)
        elif (len(args)==1) and (isinstance(args[0], Vec3d)):
             self._init_1(*args)
        elif (len(args)==1) and (isinstance(args[0], list)):
             self._init_2(float(args[0][0]), float(args[0][1]), float(args[0][2]))
        elif (len(args)==3) and (isinstance(args[0], (float,int))) and (isinstance(args[1], (float,int))) and (isinstance(args[2], (float,int))):
             self._init_2(*args)
        else:
               raise Exception('can not handle type of %s' % (args,))
 
    def __setitem__(self, index, value):
        assert isinstance(index, (int,long)), 'arg index has a wrong type. int is expected instead of '+str(type(index))
        assert isinstance(value, (float, int)), 'arg value has a wrong type. int or float expected instead of '+str(type(value))
    
        if index >= 3 or index < 0:
                raise IndexError("Index ["+str(index)+"] is invalid as it should lie in the [0,3] interval.")
    
        deref(self.inst.get())[(<int>index)] = (<double>value)
    
    def __getitem__(self,  in_0 ):
        assert isinstance(in_0, (int, long)), 'arg in_0 wrong type'
        if in_0 >= 3 or in_0 < 0:
                raise Exception("Index is too big for this vector of size 3")
        
        cdef long _idx = (<int>in_0)
        cdef double _r = deref(self.inst.get())[(<int>in_0)]
        py_result = <double>_r
        return py_result
        
    def __len__(self):
        return 3
            
    def y(self):
        cdef double _r = self.inst.get().y()
        py_result = <double>_r
        return py_result
    
    def x(self):
        cdef double _r = self.inst.get().x()
        py_result = <double>_r
        return py_result
    
    def z(self):
        cdef double _r = self.inst.get().z()
        py_result = <double>_r
        return py_result 
    
    def xy(self):
        return [<double>self.inst.get().x(), <double>self.inst.get().y()]
    
    def xyz(self):
        return [<double>self.inst.get().x(), <double>self.inst.get().y(), <double>self.inst.get().z()]
        
    def __str__(self):
        return "({:2.2f}, {:2.2f}, {:2.2f})".format(self.x(),self.y(), self.z())

    def distanceTo(self, Vec3d aPoint):
        return (self-aPoint).norm() 

