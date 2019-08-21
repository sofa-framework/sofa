# -*- coding: utf-8 -*-
"""
Numerics components we often use.

**Content:**

.. autosummary::

    RigidDof
    Vec3
    Quat
    Matrix

splib.numerics.RigidDof
***********************
.. autofunction:: RigidDof

splib.numerics.Vec3
*******************
.. autoclass:: Vec3
   :members:
   :undoc-members:

splib.numerics.Quat
*******************
.. autoclass:: Quat
   :members:
   :undoc-members:

splib.numerics.Matrix
*********************
.. autoclass:: Matrix
   :members:
   :undoc-members:

"""
from math import *
import numpy
import numpy.linalg
import SofaPython.Quaternion as Quaternion
from SofaPython.Quaternion import from_euler, to_matrix
from math import pi
from vec3 import *
from quat import *
from matrix import *

RigidDofZero = [0.0,0.0,0.0,0.0,0.0,0.0,1.0]

def to_radians(v):
    """Converts degree to radians
    
       :param v: the angle to convert
    """
    if isinstance(v, list):
        p = []
        for tp in v:
            p.append( tp * pi * 2.0 / 360.0 )
        return p
    return v * pi * 2.0 / 360.0

def TRS_to_matrix(translation, rotation=None, scale=None, eulerRotation=None):
    t = numpy.identity(4)
    s = numpy.identity(4)
    if eulerRotation != None:
        rotation = from_euler( to_radians( eulerRotation ) )

    if scale == None:
        scale = [1.0,1.0,1.0]

    r = to_matrix( rotation )

    rr = numpy.identity(4)
    rr[0:3, 0:3] = r

    t[0,3]=translation[0]
    t[1,3]=translation[1]
    t[2,3]=translation[2]

    s[0,0]=scale[0]
    s[1,1]=scale[1]
    s[2,2]=scale[2]

    return numpy.matmul( numpy.matmul(t,rr), s )

def transformPositions(position, translation=[0.0,0.0,0.0], rotation=[0.0,0.0,0.0,1.0], eulerRotation=None, scale=[1.0,1.0,1.0]):

    trs = TRS_to_matrix(translation=translation, rotation=rotation, eulerRotation=eulerRotation, scale=scale)
    tp = []
    for point in position:
        tp.append(transformPosition(point, trs).tolist())

    return tp

def transformPosition(point, matrixTRS):

    if len(point) != 3:
        raise Exception('A Point is defined by 3 coordinates [X,Y,Z] , point given : '+str(point))

    elif all(isinstance(n, int) or isinstance(n, float) for n in point):
        np = numpy.matmul( matrixTRS, numpy.append(point,1.0) )
        tp = np[0:3]

    else :
        raise Exception('A Point is a list/array of int/float, point given : '+str(point))


    return tp

class RigidDof(object):
    """Wrapper toward a sofa mechanicalobject template<rigid> as a rigid transform composed of
       a position and an orientation.

       Examples::
           
            r = RigidDof( aMechanicalObject )
            r.translate( ( r.forward * 0.2 ) )
            r.position = Vec3.zero
            r.orientation = Quat.unit
            r.rest_position = Vec3.zero
            r.rest_orientation = Quat.unit
    """
    def __init__(self, rigidobject):
        #self.rigidobject.init()
        self.rigidobject = rigidobject

    def getPosition(self, index=0, field="position"):   
        return self.rigidobject.getData(field).value[index][:3]

    def setPosition(self, v, field="position"):
        p = self.rigidobject.getData(field)
        if(not isinstance(v,list)):
            v = list(v)
        p.value = v + p.value[0][3:]
        
    position = property(getPosition, setPosition)

    def setOrientation(self, q, field="position"):
        p = self.rigidobject.getData(field)
        if(not isinstance(v,list)):
           v = list(v)
        p.value = p.value[0][:3] + v

    def getOrientation(self, field="position"):
        return self.rigidobject.getData(field).value[0][3:]
    orientation = property(getOrientation, setOrientation)

    def getForward(self, field="position"):
        o = self.rigidobject.getData(field).value[0][3:]
        return numpy.matmul(TRS_to_matrix([0.0,0.0,0.0], o), numpy.array([0.0,0.0,1.0,1.0]))[:3]
    forward = property(getForward, None)

    def getLeft(self, field="position"):
        o = self.rigidobject.getData(field).value[0][3:]
        return numpy.matmul(TRS_to_matrix([0.0,0.0,0.0], o), numpy.array([1.0,0.0,0.0,1.0]))[:3]
    left = property(getLeft, None)

    def getUp(self, field="position"):
        o = self.rigidobject.getData(field).value[0][3:]
        return numpy.matmul(TRS_to_matrix([0.0,0.0,0.0], o), numpy.array([0.0,1.0,0.0,1.0]))[:3]
    up = property(getUp, None)

    def copyFrom(self, t, field="position"):
        self.rigidobject.getData(field).value = t.rigidobject.getData(field).value

    def translate(self, v, field="position"):
        p = self.rigidobject.getData(field)
        to = p.value[0]
        t = Transform(to[:3], orientation=to[3:])
        t.translate(v)
        p.value = t.toSofaRepr()

    def rotateAround(self, axis, angle, field="position"):
        p = self.rigidobject.getData(field)
        pq = p.value[0]
        p.value =  pq[:3] + list(Quaternion.prod(axisToQuat(axis, angle), pq[3:]))

    def __getattr__(self, key):
        if key in self.__dict__:
                return self.__dict__[key]                        
        if key in ["rest_position", "position"]:
                return self.getPosition(field=key)     
        elif key in ["orientation", "rest_orientation"]:
                return self.getOrientation(field=key)        
        return self.rigidobject.getData(key).value          

    def __setattr__(self, key,value):
        if key in ["rest_position", "position"]:
                return self.setPosition(value, field=key)     
        elif key in ["orientation", "rest_orientation"]:
                return self.setOrientation(value, field=key)        
        elif key == "rigidobject":
                self.__dict__[key] = value   
        else:
                self.rigidobject.getData(key).value = list(value)
                
class Transform(object):
    def __init__(self, translation, orientation=None, eulerRotation=None):
        self.translation = translation
        if eulerRotation != None:
            self.orientation = from_euler( to_radians( eulerRotation ) )
        elif orientation != None:
            self.orientation = orientation
        else:
            self.orientation = [0,0,0,1]

    def translate(self, v):
        self.translation = vadd(self.translation, v)
        return self

    def toSofaRepr(self):
            return self.translation + list(self.orientation)

    def getForward(self):
        return numpy.matmul(TRS_to_matrix([0.0,0.0,0.0], self.orientation), numpy.array([0.0,0.0,1.0,1.0]))

    forward = property(getForward, None)

def getOrientedBoxFromTransform(translation=[0.0,0.0,0.0], rotation=[0.0,0.0,0.0,1.0], eulerRotation=None, scale=[1.0,1.0,1.0]):
        # BoxROI unitaire
        pos = [[-0.5, 0.0,-0.5],
               [-0.5, 0.0, 0.5],
               [ 0.5, 0.0, 0.5]]

        if eulerRotation is not None:
            rotation=eulerRotation

        depth = [scale[1]]
        return transformPositions(position=pos, translation=translation, rotation=rotation, eulerRotation=eulerRotation, scale=scale) + depth



def axisToQuat(axis, angle):
    na  = numpy.zeros(3)
    na[0] = axis[0]
    na[1] = axis[1]
    na[2] = axis[2]
    return list(Quaternion.axisToQuat(na, angle))
    
def createScene(rootNode):
        import Sofa
        d = rootNode.createObject("MechanicalObject", template="Rigid3", name="dofs", position=[0,0,0,0,0,0,1])
        r = RigidDof(d)
        print(str(r.position))
        print(str(r.translate([1,2,3])))
        print(str(r.position))
        print(str(r.translate([1,2,3])))
        
        print(str(r.getPosition()))
        print(str(r.getPosition(field="rest_position")))
        print(str(r.getOrientation()))
        print(str(r.getOrientation(field="rest_position")))

