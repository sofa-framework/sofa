import Sofa
from SofaPython import Quaternion as quat

import numpy as np

np.set_string_function( lambda x: ' '.join( map(str, x)), repr = False )

import path
import tool

# rigid body operations
def id():
    res = np.zeros(7)
    res[-1] = 1
    return res

def inv(x):
    res = np.zeros(7)

    res[3:] = quat.conj(x[3:])
    res[:3] = -quat.rotate(res[3:], x[:3])
    
    return res

def prod(x, y):
    print x.inv()
    res = np.zeros(7)

    res[:3] = x[:3] + quat.rotate(x[3:], y[:3])
    res[3:] = quat.prod(x[3:], y[3:])

    return res



class DOFs(object):

    def __init__(self, arg, **kwargs):
        if type(arg) == Sofa.BaseMechanicalState:
            self.obj = arg
        else:
            node = arg
            kwargs.setdefault('template', 'Rigid')

            pos = np.zeros(7)
            pos[-1] = 1
            kwargs.setdefault('position', pos)

            self.obj = node.createObject('MechanicalObject',
                                         name = 'dofs',
                                         **kwargs)

    @property
    def position(self):
        return np.array(self.obj.position[0])

    @position.setter
    def position(self, value):
        self.obj.position = str(value)


    @property
    def node(self):
        return self.obj.getContext()
    
    @property
    def center(self):
        return np.array(self.obj.position[0][:3])

    @center.setter
    def center(self, value):
        self.obj.position = str(np.hstack( (value, self.obj.position[0][3:])) )
        
    @property
    def orient(self):
        return np.array(self.obj.position[0][3:])

    @orient.setter
    def orient(self, value):
        self.obj.position = str(np.hstack( (self.obj.position[0][:3]. value) ))


    def map_vec3(self, name, position):
        # note: you may map multiple points at once
        
        node = self.obj.getContext()
        
        res = node.createChild(name)
        res.createObject('MechanicalObject',
                         template = 'Vec3',
                         name = 'dofs',
                         position = position,
                         velocity = np.zeros( len(position) ) )
        res.createObject('RigidMapping')
        return res

    def map_rigid(self, name, position):
        # note: only one rigid for now
        
        node = self.node.createChild(name)

        res = DOFs(node)
        
        node.createObject('AssembledRigidRigidMapping',
                          source = '0 ' + str(position) )
        
        return res


    def map_relative(self, node, child, **kwargs):

        log = kwargs.get('log', False)

        template = 'Vec6' if log else 'Rigid'
        
        dofs = DOFs(node) if not log else node.createObject('MechanicalObject',
                                                            name = 'dofs',
                                                            template = template,
                                                            position = np.zeros(6) )

        obj = dofs.obj if not log else dofs
        
        input = [ '@' + path.relative(obj, x) for x in [self.obj, child.obj] ]
        
        node.createObject('RigidJointMultiMapping',
                          template = 'Rigid,' + template,
                          name = 'mapping',
                          input = ' '.join(input),
                          output = '@dofs',
                          pairs = '0 0')
                          
        return dofs


class Mass(object):
    
    def __init__(self, node, **kwargs):

        kwargs.setdefault('template', 'Rigid')

        mass = kwargs.setdefault('mass', 1.0)
        kwargs.setdefault('inertia', mass * np.ones(3))

        self.obj = node.createObject('RigidMass',
                                     name = 'mass',
                                     **kwargs)
        

    @property
    def inertia(self):
        return np.array( self.obj.inertia )

    @inertia.setter
    def inertia(self, value):
        self.obj.inertia = str(value)

    @property
    def mass(self):
        return self.obj.mass[0][0]

    @mass.setter
    def mass(self, value):
        self.obj.mass = value
    

    def box(self, sx, sy, sz, rho = 1000.0):
        # TODO
        volume = sx * sy * sz 
        self.mass = rho * volume

        dim = np.array([sx, sy, sz], float)
        dim2 = dim * dim
        
        self.inertia = self.mass / 12.0 * (dim2[ [1, 2, 0] ] + dim2[ [2, 0, 1] ])
        
    def cylinder(self, radius, length, rho = 1000.0):
        # TODO axis...
        raise Exception('unimplemented')

    def mesh(self, filename, rho = 1000.0):
        # TODO
        raise Exception('unimplemented')
     


class Body(object):

    def __init__(self, node, name, **kwargs):

        self.node = node.createChild(name)

        self.dofs = DOFs(self.node, **kwargs)
        self.mass = Mass(self.node, **kwargs)

    @property
    def name(self):
        return self.node.name

    @name.setter
    def name(self, value):
        self.node.name  = value
    

    @property
    def fixed(self):
        return self.node.getObject('fixed') is not None

    @fixed.setter
    def fixed(self, value):
        has = self.node.getObject('fixed')
        if has and not value: self.node.removeObject(has)
        if value and not has: self.node.createObject('FixedConstraint',
                                                     name = 'fixed',
                                                     indices = 0)



class Collision(object):

    def __init__(self, node, filename, **kwargs):
        self.node = node.createChild('collision')
        
        self.loader = self.node.createObject('MeshObjLoader',
                                             name = 'loader',
                                             filename = filename,
                                             **kwargs)
        topology = self.node.createObject('MeshTopology',
                                          name = 'topology',
                                          src = '@loader')
        
        dofs = self.node.createObject('MechanicalObject',
                                      name = 'dofs',
                                      template = 'Vec3')
        
        triangles = self.node.createObject('TriangleModel',
                                           name = 'model')
        
        mapping = self.node.createObject('RigidMapping')

    @property
    def scale(self):
        return np.array( self.loader.scale3d )


    @scale.setter
    def scale(self, value):
        # TODO if type(value) is float
        self.loader.scaled3d = str( value )


    @property
    def translation(self):
        return np.array( self.loader.translation )

    
    @translation.setter
    def translation(self, value):
        self.loader.translation = str( value )

    # TODO rotation




class Joint(object):

    def __init__(self, dofs, parent, child):

        # nodes
        self.relative = dofs.node.createChild('relative')
        self.constraint = self.relative.createChild('constraint')

        self.parent = parent
        self.child = child
        
        relative_dofs = parent.map_relative(self.relative, child)

        dofs.map_relative(self.constraint, relative_dofs, log = True)

        self.constraint.createObject('UniformCompliance',
                                     template = 'Vec6',
                                     compliance = 0)
        
        self.constraint.createObject('Stabilization')

import mapping        
class HingeJoint(Joint):

    def __init__(self, node, parent, child):

        self.dofs = tool.dofs(node, 'Vec1')

        spring = node.createChild('spring')
        tool.dofs(spring, 'Vec1')
        spring.createObject('IdentityMapping', template = 'Vec1,Vec1')
        
        spring.createObject('UniformCompliance',
                            template = 'Vec1',
                            compliance = 1e-4,
                            isCompliance = False)

        # for some reason we need mass for stability :-/
        self.mass = node.createObject('UniformMass',
                                      template = 'Vec1',
                                      mass = 1)
        

        self.mapping = mapping.HingeJoint(self.dofs)


        dofs = DOFs( self.mapping.output )
        
        Joint.__init__(self, dofs, parent, child)
        
        
    

