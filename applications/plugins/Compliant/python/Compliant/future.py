import Sofa
import ctypes


class PyObject(ctypes.Structure):
    _fields_ = (
        ("ob_refcnt", ctypes.c_ssize_t),
        ("ob_type", ctypes.py_object),
    )

class SlotsPointer(PyObject):
    _fields_ = [('dict', ctypes.POINTER(PyObject))]


# stolen from
# http://stackoverflow.com/questions/6738987/extension-method-for-python-built-in-types
def class_dict( klass ):
    '''proxy for class __dict__ through CPython. not quite
    pythonic.'''

    name = klass.__name__
    slots = getattr(klass, '__dict__')

    pointer = SlotsPointer.from_address(id(slots))

    # not exactly sure why we need this dictionary but hey, it works
    namespace = {}

    ctypes.pythonapi.PyDict_SetItem(
        ctypes.py_object(namespace),
        ctypes.py_object(name),
        pointer.dict,
    )

    return namespace[name]
    


    
def extends(base):
    '''decorator for SofaPython class extensions. 

    use this to add methods to existing SofaPython binding classes
    from the python side.

    '''
    def res(cls):

        # TODO we should filter crap such as '__module__' 
        class_dict( base ).update( cls.__dict__ )

        # print patch.class_dict( base )
        return cls

    return res





# PySPtr and friends
class SofaObject(PyObject):
    _fields_ = [
        ('obj', ctypes.c_void_p)
    ]
    


@extends(Sofa.Base)
class Base:
    
    @property
    def _as_parameter_(self):
        '''use sofa objects in c functions as Base*'''
        return SofaObject.from_address( id(self) ).obj
    


class Plugin(object):

    def __init__(self, name):
        directory = Sofa.build_dir()
        
        import platform
        system = platform.system()
        extension = 'so'

        if system == 'Windows':
            extension = 'dll'
        elif system == 'Darwin':
            extension = 'dylib'
        
        prefix = 'lib'
        suffix = extension
        
        self.dll = ctypes.CDLL('{}/lib/{}{}.{}'.format(directory, prefix, name, suffix))

        self.dll.argtypes.argtypes = [ type(self.dll.argtypes) ]
        self.dll.argtypes.restype = ctypes.c_char_p

        self.dll.restype.argtypes = [ type(self.dll.restype) ]
        self.dll.restype.restype = ctypes.c_char_p

    def __getattr__(self, name):

        try:
            res = getattr(self.dll, name)
        except:
            print 'error: function "{}" not found in plugin'.format(name)
            raise

        try:
            if not res.argtypes:
                res.argtypes = eval(self.dll.argtypes( res ), ctypes.__dict__ )
                
            if not res.restype:
                res.restype = eval(self.dll.restype( res ), ctypes.__dict__)
                
        except TypeError:
            print 'error computing argtypes/restypes for',
            print name,
            print 'did you register it using python::add ?'
            raise
        return res
        


from types import Rigid, Quaternion

