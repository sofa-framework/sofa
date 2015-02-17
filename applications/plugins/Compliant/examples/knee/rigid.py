

import numpy as np

np.set_string_function( lambda x: ' '.join( map(str, x)), repr = False )


class DOFs(object):

    def __init__(self, node, **kwargs):
        kwargs.setdefault('template', 'Rigid')

        pos = np.zeros(7)
        pos[-1] = 1
        kwargs.setdefault('position', pos)
        
        self.obj = node.createObject('MechanicalObject',
                                     name = 'dofs',
                                     **kwargs)

    @property
    def center(self):
        return self.obj.position[0][:3]

    @center.setter
    def center(self, value):
        self.obj.position = str(np.hstack( (value, self.obj.position[0][3:])) )
        
    @property
    def orient(self):
        return self.obj.position[0][3:]

    @orient.setter
    def orient(self, value):
        self.obj.position = str(np.hstack( (self.obj.position[0][:3]. value) ))


    def map_vec3(self, name, position):
        node = self.obj.getContext()
        
        res = node.createChild(name)
        res.createObject('MechanicalObject',
                         template = 'Vec3',
                         name = 'dofs',
                         position = position)
        res.createObject('RigidMapping')
        return res
    

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

        dim = np.array([sx, sy, sz])
        dim2 = dim * dim
        
        self.inertia = self.mass / 12.0 * dim2[ [1, 0, 0] ] + dim2[ [2, 2, 1] ]
        
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
