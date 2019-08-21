'''easy python mappings'''

import numpy as np
import script
import path
import tool

class Script(script.Controller):

    def __new__(cls, node, instance):
        res = script.Controller.__new__(cls, node)
        res.instance = instance

        # callbacks
        res.cb = []
        
        return res
    

    def update(self):
        self.instance.update()

        # FIXME mapping gets updated correctly, but output positions
        # are lagging one step behind, so we need to push manually
        # until a better solution is found
        self.instance.output.position = str(self.instance.value)
        
        for cb in self.cb: cb()
        
    def onBeginAnimationStep(self, dt):
        self.update()
        
    def reset(self):
        self.update()
        

class Base(object):
    
    def __init__(self, node, name, dim, **kwargs):

        self.node = node.createChild(name)
        
        self.script = Script(self.node, self)

        self.template = kwargs['template'].split(',')

        self.output = tool.dofs(self.node, self.template[-1], dim)
        
        self.input = kwargs['input']
        input = ' '.join( [ '@' + path.relative(self.output, x)
                            for x in self.input] )

        self.update_size()
        self.value_size = dim * tool.coord_size(self.template[-1])

        self.mapping = self.node.createObject('PythonMultiMapping',
                                              name = 'mapping',
                                              template = ','.join(self.template),
                                              input = input,
                                              output = '@dofs',
                                              jacobian = np.zeros( (self.rows, self.cols)),
                                              value = np.zeros( self.value_size ))


    def update_size(self):
        self.rows = tool.matrix_size( self.output )
        self.cols = sum( map(tool.matrix_size, self.input) )
        
        self.size = (self.rows, self.cols)

        
    @property
    def jacobian(self):
        res = np.array(self.mapping.jacobian)
        return res.reshape( (self.rows, self.cols) )

    @property
    def value(self):
        return np.array(self.mapping.value)
    
    @jacobian.setter
    def jacobian(self, value):
        self.mapping.jacobian = str(value.flatten())

        
    @value.setter
    def value(self, value):
        self.mapping.value = str(value)


import math

class PointPlaneDistance(Base):

    def __init__(self, node, plane, point, **kwargs):

        self.plane_indices = kwargs.get('plane_indices', [0, 1, 2])
        self.point_index = kwargs.get('point_index', 0)

        self.plane_dofs = plane
        self.point_dofs = point
        
        self.offset = kwargs.get('offset', 0)
        
        Base.__init__(self, node, 'point-plane-distance', 1,
                        input = [plane, point],
                        template = 'Vec3,Vec1')


        self.sign = 1
        
    def update(self):
        A = np.zeros( (3, 3) )

        plane = [self.plane_dofs.position[i] for i in self.plane_indices ]
        point = self.point_dofs.position[ self.point_index ]

        # TODO constant offset on both sides to avoid singular A
        offset = np.zeros(3)
        
        for i in xrange(3):
            A[:, i] = plane[i] + offset

        b = point + offset

        Ainv = np.linalg.inv(A)
        
        u = Ainv.transpose().dot( np.ones(3) )
        alpha = math.sqrt( u.dot(u) )
        
        self.value = self.sign * (u.dot(b) - self.offset * alpha - 1)

        self.update_size()

        J = np.zeros( (self.rows, self.cols) )

        right = tool.matrix_size( self.point_dofs )
        left = tool.matrix_size( self.plane_dofs )
        
        # point part
        J[:, -right:][:, 3 * self.point_index : 3 * self.point_index + 3] = u.transpose()

        # plane part
        n = u / alpha

        v = b - self.offset * n

        w = -Ainv.dot(v)

        for i, p in enumerate(self.plane_indices):
            J[:, :-right][:, 3 * p:3*p + 3] = w[i] * u.transpose()

        self.jacobian = self.sign * J

        # print 'jacobian', self.jacobian
        # print 'value', self.value



import rigid
from SofaPython import Quaternion as quat

class HingeJoint(Base):

    def __init__(self, dofs, **kwargs):

        node = dofs.getContext()

        self.axis = kwargs.get('axis', np.array([1, 0, 0]))
        self.dofs = dofs
        
        Base.__init__(self, node, 'joint', 1,
                      input = [dofs],
                      template = 'Vec1,Rigid')

        # print self.script.instance

    def update(self):

        value = rigid.id()
        value[3:] = quat.exp( self.dofs.position[0][0] * self.axis )
            
        self.value = value

        jacobian = np.zeros( self.size )
        jacobian[3:, :] = self.axis.reshape( (3, 1) )

        self.jacobian = jacobian

        # print self.value
        # print self.jacobian


