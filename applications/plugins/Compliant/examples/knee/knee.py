
import Sofa

import os

import numpy as np

import rigid
import mapping
import script
import tool

from tool import vec


def setup(node, **kwargs):

    tool.require(node, 'Compliant')

    use_pouf = False

    if not use_pouf:
        ode = node.createObject('CompliantImplicitSolver', **kwargs)
        num = node.createObject('SequentialSolver', **kwargs)
    else:
        tool.require(node, 'pouf')    
        ode = node.createObject('pouf.solver', **kwargs)
        num = node.createObject('pouf.pgs', **kwargs)

        ode.stabilization = True
        num.nlnscg = True

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

    

class Controller(script.Controller):

    def reset(self):
        print 'reset'


def createScene( node ):
    reload(rigid)
    reload(script)
    import path
    reload(path)
    
    setup( node, iterations = 1000, precision = 1e-8,
           displayFlags = 'showMapping showBehavior showCollisionModels' )
    
    scene = node.createChild('scene')
    
    femur = rigid.Body(scene, 'femur', draw = True)
    femur.length = 0.6
    femur.width = 0.05
    
    femur.dofs.center = [0.0, femur.length / 2, 0.0]

    rho = 1800
    
    femur.mass.box(femur.width, femur.length, femur.width, rho)

    radius = 0.05
    w, h = 2 * radius, - femur.length / 2.0

    condyles = femur.dofs.map_vec3('condyles', vec( w, h, 0,
                                                    -w, h, 0) )

    condyles.createObject('SphereModel', radius = radius)

    tibia = rigid.Body(scene, 'tibia', draw = True)
    tibia.length = 0.6
    tibia.width = 0.05
    
    tibia.dofs.center = [0, -tibia.length / 2, 0]
    tibia.mass.box(tibia.width, tibia.length, tibia.width, rho)

    mesh = os.path.join(os.path.dirname(__file__), 'box-inner.obj')

    size = w + radius

    tibia.collision = rigid.Collision(tibia.node,
                                      mesh,
                                      scale3d = vec(size, 2 * radius, radius),
                                      translation = vec(0, -h + radius, 0))

    femur.fixed = True


    point = femur.dofs.map_vec3('e', vec(0, 0.1, 0) )
    

    femur.attach = femur.dofs.map_vec3('ligament', vec(0, h, 0))
    tibia.attach = tibia.dofs.map_vec3('ligament', vec(0, 0.9 * tibia.length / 2, 0))

    delta = scene.createChild('delta')
    delta.createObject('MechanicalObject',
                       template = 'Vec3',
                       name = 'dofs',
                       position = np.zeros(3))
    
    delta.createObject('DifferenceMultiMapping',
                       template = 'Vec3,Vec3',
                       input = '@../femur/ligament/dofs @../tibia/ligament/dofs',
                       pairs = '0 0',
                       output = '@dofs')

    stiffness = 1e7
    ff = delta.createObject('UniformCompliance',
                            template = 'Vec3',
                            compliance = 1.0 / stiffness)


    # obj = node.createObject('PythonScriptController',
    #                         filename = __file__,
    #                         classname = 'Bob')

    test = mapping.Affine(scene, 'test', 1,
                          template = 'Vec3',
                          input = [ path.get_object(femur.node, '@ligament/dofs' ) ] )


    
    
    return 0

    
