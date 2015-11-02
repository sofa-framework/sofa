import SofaPython.Quaternion as quat
import Tools
from Tools import cat as concat
from numpy import *

class Frame:
        # a rigid frame, group operations are available.

        # TODO kwargs
        def __init__(self, value = None):
                if value is not None:
                        self.translation = value[:3]
                        self.rotation = value[3:]
                else:
                        self.translation = [0, 0, 0]
                        self.rotation = [0, 0, 0, 1]

        def insert(self, parent, template='Rigid', **args):
                self.node = parent
                return parent.createObject('MechanicalObject',
                                           template = template,
                                           position = str(self),
                                           **args)

        def __str__(self):
                return concat(self.translation) + ' ' + concat(self.rotation)

        def copy(self):
                # TODO better !
                return Frame().read( str(self) )

        def setValue(self, v):
             self.translation = v[:3]
             self.rotation = v[3:]
             return self

        def read(self, str):
                num = map(float, str.split())
                self.translation = num[:3]
                self.rotation = num[3:]
                return self

        def offset(self):
            return hstack((self.translation, self.rotation))

        def __mul__(self, other):
            res = Frame()
            res.translation = self.translation + quat.rotate(self.rotation, other.translation)
            res.rotation = quat.prod( self.rotation, other.rotation)

            return res

        def inv(self):
            res = Frame()
            res.rotation = quat.conj( self.rotation )
            res.translation = - quat.rotate(res.rotation, self.translation)
            return res

        def set(self, **kwargs):
                for k in kwargs:
                        setattr(self, k, kwargs[k])

                return self

        # TODO more: wrench/twist frame change.

        def apply(self, vec):
            """ apply transformation to vec (a [x,y,z] vector)
            return the result
            """
            return array(quat.rotate(self.rotation, vec) + asarray(self.translation))

        def applyInv(self, vec):
            """ apply the inverse transformation to vec (a [x,y,z] vector)
            return the result
            """
            return self.inv().apply(vec)

        def __eq__(self, other):
            """ floating point comparison """
            return allclose(self.translation, other.translation) and allclose(self.rotation, other.rotation)

        def __ne__(self, other):
            """ floating point comparison """
            return not allclose(self.translation, other.translation) or not allclose(self.rotation, other.rotation)
