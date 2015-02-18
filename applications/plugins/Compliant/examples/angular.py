import Sofa

import numpy as np
import math
from SofaPython import Quaternion as quat

# use numpy vectors directly (watch out, string conversion might be
# lossy)

np.set_string_function( lambda x: ' '.join( map(str, x)),
                        repr=False )

def createScene(node):

    node.createObject('RequiredPlugin',
                      pluginName = 'Compliant')


    ode = node.createObject('CompliantImplicitSolver')
    num = node.createObject('SequentialSolver')

    # ode.debug = 1
    node.dt = 0.01

    pos = np.zeros(7)
    vel = np.zeros(6)
    force = np.zeros(6)

    alpha = math.pi / 4.0
    
    q = quat.exp([0, 0, alpha])

    pos[:3] = [-0.5, 0, 0]
    pos[3:] = q

    mass = 1.0

    # change this for more fun
    dim = np.array([1, 2, 1])

    dim2 = dim * dim
    
    
    inertia = mass / 12.0 * (dim2[ [1, 2, 0] ] + dim2[ [2, 0, 1] ])
    volume = 1.0
    
    force[3:] = quat.rotate(q, [0, 1, 0])

    scene = node.createChild('scene')
    
    good = scene.createChild('good')
    
    dofs = good.createObject('MechanicalObject',
                             template = 'Rigid',
                             name = 'dofs',
                             position = pos,
                             velocity = vel,
                             showObject = 1)

    good.createObject('RigidMass',
                      template = 'Rigid',
                      name = 'mass',
                      mass = mass,
                      inertia = inertia)

    good.createObject('ConstantForceField',
                      template = 'Rigid',
                      name = 'ff',
                      forces = force)

    bad = scene.createChild('bad')

    pos[:3] = [0.5, 0, 0]
    dofs = bad.createObject('MechanicalObject',
                            template = 'Rigid',
                            name = 'dofs',
                            position = pos,
                            velocity = vel,
                            showObject = 1)

    inertia_matrix = np.diag(inertia)
    
    def cat(x): return ' '.join( map(str, x))

    def print_matrix(x):
        return '[' + ','.join(map(str, x)) + ']'

    
    bad.createObject('UniformMass',
                     template = 'Rigid',
                     name = 'mass',
                     mass = cat([mass, volume, print_matrix(inertia_matrix / mass)]))
    
    bad.createObject('ConstantForceField',
                      template = 'Rigid',
                      name = 'ff',
                      forces = force)

                
    node.gravity = '0 0 0'
