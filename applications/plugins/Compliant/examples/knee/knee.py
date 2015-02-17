
import Sofa

import numpy as np
import os

np.set_string_function( lambda x: ' '.join( map(str, x)), repr = False )

import rigid

def require(node, plugin):
    return node.createObject('RequiredPlugin', pluginName = plugin)


def setup(node, **kwargs):

    require(node, 'Compliant')

    use_pouf = False

    if not use_pouf:
        ode = node.createObject('CompliantImplicitSolver', **kwargs)
        num = node.createObject('SequentialSolver', **kwargs)
    else:
        require(node, 'pouf')    
        # ode = node.createObject('pouf.solver', **kwargs)
        # num = node.createObject('pouf.pgs', **kwargs)

        # ode.stabilization = True
        # num.nlnscg = True

    style = node.createObject('VisualStyle', **kwargs)
    node.createObject('DefaultPipeline', name = 'pipeline')
    node.createObject('BruteForceDetection', name = 'detection')

    proximity = node.createObject('NewProximityIntersection',
                                  name = 'proximity' )

    proximity.alarmDistance = 0.1
    proximity.contactDistance = 0.00

    manager = node.createObject('DefaultContactManager',
                                name = 'manager',
                                response = "CompliantContact",
                                responseParams = "compliance=0" )

    
def vec(*args):
    return np.array( args, float )
    
def createScene( node ):
    reload(rigid)

    setup( node, iterations = 10000, precision = 0,
           displayFlags = 'showMapping showBehavior showCollisionModels' )
    
    scene = node.createChild('scene')
    
    femur = rigid.Body(scene, 'femur', showObject = True)
    femur.length = 0.6
    femur.width = 0.05
    
    femur.dofs.center = [0.0, femur.length / 2, 0.0]

    rho = 1800
    
    femur.mass.box(femur.width, femur.length, femur.width, rho)

    radius = 0.05
    x, y = 2 * radius, - 0.3

    condyles = femur.dofs.map_vec3('condyles', vec( x, y, 0,
                                                    -x, y, 0) )

    condyles.createObject('SphereModel', radius = radius)

    tibia = rigid.Body(scene, 'tibia', showObject = True)
    tibia.length = 0.6
    tibia.width = 0.05
    
    tibia.dofs.center = [0, -tibia.length / 2, 0]
    tibia.mass.box(tibia.width, tibia.length, tibia.width, rho)

    mesh = os.path.join(os.path.dirname(__file__), 'box-inner.obj')

    size = x + radius

    tibia.collision = rigid.Collision(tibia.node,
                                      mesh,
                                      scale3d = vec(size, 2 * radius, radius),
                                      translation = vec(0, -y + radius, 0))

    femur.fixed = True

    femur.attach = femur.dofs.map_vec3('attach', vec(0, y, 0))
    tibia.attach = tibia.dofs.map_vec3('attach', vec(0, 0.9 * tibia.length / 2, 0))

    delta = scene.createChild('delta')
    delta.createObject('MechanicalObject',
                       template = 'Vec3',
                       name = 'dofs',
                       position = np.zeros(3))
    
    delta.createObject('DifferenceMultiMapping',
                       template = 'Vec3,Vec3',
                       input = '@../femur/attach/dofs @../tibia/attach/dofs',
                       pairs = '0 0',
                       output = '@dofs')

    stiffness = 1e8
    ff = delta.createObject('UniformCompliance',
                            template = 'Vec3',
                            compliance = 1.0 / stiffness)

    return 0

    
