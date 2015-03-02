
import patch

type_dict = {}

def cast( obj ):
    '''cast an object into its specialized handling class, as decorated by
    sofa_object.

    '''

    # TODO fancy template regexp ?
    try:
        cls = type_dict[obj.getClassName()][obj.getTemplateName()]
        return patch.instance(obj, cls)
    except AttributeError:
        return obj



def extends(base):
    '''decorator for SofaPython class extensions. 

    use this to add methods to existing SofaPython binding classes
    from the python side.

    '''
    def res(cls):
        patch.class_dict( base ).update( cls.__dict__ )
        return cls

    return res



def sofa_class(classname,
               template = ''):
    '''decorator for specialized classes.

    use this to define specialized classes for handling
    e.g. MechanicalObject<Vec3d>. 

    objects obtained through Node.createObject or Node.getObject will
    be wrapped according to their classname/templatename.

    '''
    
    def res( cls ):
        type_dict.setdefault(classname, {})[template] = cls
        return cls

    return res

# actual wrapping starts here
# should probably go somewhere else

import Sofa

@extends(Sofa.Node)
class Node:
    _createObject = Sofa.Node.createObject
    _getObject = Sofa.Node.getObject

    def createObject(self, typename, **kwargs):
        return cast( Node._createObject(self, typename, **kwargs) )

    def getObject(self, name):
        return cast( Node._getObject(self, name) )
    


@extends(Sofa.BaseMechanicalState)
class BaseMechanicalState:

    _getattr = Sofa.Base.__getattribute__
    _setattr = Sofa.Base.__setattr__

    def __getattribute__(self, name):
        # overrrides data access
        if name in BaseMechanicalState.__dict__:
            return object.__getattribute__(self, name)
        else:
            return BaseMechanicalState._getattr(self, name)

    def __setattr__(self, name, value):
        # overrrides data access
        if name in BaseMechanicalState.__dict__:
            return BaseMechanicalState.__dict__[name].__set__(self, value)
        else:
            return BaseMechanicalState._setattr(self, name, value)

        
    # wrap state vectors as numpy arrays
    @property
    def position(self):
        return np.array( BaseMechanicalState._getattr(self, 'position') )


    @position.setter
    def position(self, value):
        BaseMechanicalState._setattr(self, 'position', str(value))


    @property
    def velocity(self):
        return np.array( BaseMechanicalState._getattr(self, 'velocity') )


    @velocity.setter
    def velocity(self, value):
        BaseMechanicalState._setattr(self, 'velocity', str(value))


    @property
    def force(self):
        return np.array( BaseMechanicalState._getattr(self, 'force') )

    @force.setter
    def force(self, value):
        BaseMechanicalState._setattr(self, 'force', str(value))
    

import numpy as np

# magic function to get sofa-readable string output for numpy arrays
np.set_string_function( lambda x: ' '.join( map(str, x)), repr = False )


@sofa_class('MechanicalObject', 'Rigid3f')
@sofa_class('MechanicalObject', 'Rigid3d')
class Rigid3Object(Sofa.BaseMechanicalState):

    def __new__(cls, node, **kwargs):

        kwargs.setdefault('position', Rigid3())
        kwargs.setdefault('template', 'Rigid')
        kwargs.setdefault('name', 'dofs')
        
        return node.createObject('MechanicalObject',
                                 **kwargs)
    
    @property
    def position(self):
        '''a view on rigid positions'''
        return super(Rigid3Object, self).position.view(dtype = Rigid3)





# datatypes

class Rigid3(np.ndarray):

    @property
    def center(self):
        return self[:3]

    @center.setter
    def center(self, value):
        self[:3] = value

    @property
    def orient(self):
        return self[3:].view(Quaternion)

    @orient.setter
    def orient(self, value):
        self[3:] = value

    def __new__(cls, *args):
        return np.ndarray.__new__(cls, 7)
        
    def __init__(self):
        self[-1] = 1
        self[:6] = 0

    def inv(self):
        res = Rigid3()
        res.orient = self.orient.inv()
        res.center = -res.orient.rotate(self.center)
        return res

    def __mul__(self, other):
        res = Rigid3()

        res.orient = self.orient * other.orient
        res.center = self.center + self.orient.rotate(other.center)
        
        return res


    def __call__(self, x):
        '''applies rigid transform to vector x'''
        return self.center + self.orient(x)
    

import math
import sys

class Quaternion(np.ndarray):
    def __new__(cls, *args):
        return np.ndarray.__new__(cls, 4)
        
    def __init__(self):
        '''identity quaternion'''
        self.real = 1
        self.imag = 0
        
    def inv(self):
        '''inverse'''
        return self.conj() / self.dot(self)
    
    def conj(self):
        '''conjugate'''
        res = Quaternion()
        res.real = self.real
        res.imag = -self.imag

        return res

    @property
    def real(self): return self[-1]

    @real.setter
    def real(self, value): self[-1] = value

    @property
    def imag(self): return self[:3]

    @imag.setter
    def imag(self, value): self[:3] = value

    def normalize(self):
        '''normalize quaternion'''
        self /= math.sqrt( self.dot(self) )

    def flip(self):
        '''flip quaternion in the real positive halfplane, if needed'''
        if self.real < 0: self = -self

    def __mul__(self, other):
        '''quaternion product'''
        res = Quaternion()
        res.real = self.real * other.real - self.imag.dot(other.imag)
        res.imag = self.real * other.imag + other.real * self.imag + np.cross(self.imag, other.imag)
        
        return res
         

    def __call__(self, x):
        '''rotate a vector. self should be normalized'''
        
        tmp = Quaternion()
        tmp.real = 0
        tmp.imag = x

        return (self * tmp * self.conj()).imag
    
    @staticmethod
    def exp(x):
        '''quaternion exponential (doubled)'''

        theta = np.linalg.norm(x)

        res = Quaternion()
        
        if math.fabs(theta) < sys.float_info.epsilon:
            return res

        s = math.sin(theta / 2.0)
        c = math.cos(theta / 2.0)

        res.real = c
        res.imag = x * (s / theta)

        return res
