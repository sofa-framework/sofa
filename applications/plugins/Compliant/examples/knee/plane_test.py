

import Sofa
import mapping
import rigid
import tool
import constraint

from tool import vec

import script
import numpy as np

reload(script)
reload(rigid)
reload(mapping)

def setup(node, **kwargs):
    
    tool.require(node, 'Compliant')
    
    ode = node.createObject('CompliantImplicitSolver', **kwargs)
    num = node.createObject('SequentialSolver', **kwargs)
    # num = node.createObject('LDLTSolver', kkt = True, **kwargs)

    # tool.require(node, 'pouf')    
    # ode = node.createObject('pouf.solver', stabilization = True, **kwargs)
    # num = node.createObject('pouf.pgs', **kwargs)
    
    ode.debug = 2

    style = node.createObject('VisualStyle', **kwargs)

def createScene(node):
    
    setup(node, precision = 0, iterations = 100,
          displayFlags = 'showMapping showBehavior showCollisionModels' )

    scene = node.createChild('scene')

    body = rigid.Body(scene, 'body', draw = True)


    plane = body.dofs.map_vec3('plane', vec(1, 1, 0,
                                            0, 1, 0,
                                            0, 1, 1))

    plane.createObject('UniformMass', template = 'Vec3', mass = 1)

    point = scene.createChild('point')
    dofs = point.createObject('MechanicalObject',
                              template = 'Vec3',
                              name = 'dofs',
                              position = '1 2 1')

    point.createObject('UniformMass', template = 'Vec3', mass = 1)
    point.createObject('SphereModel', radius = 0.01)

    test = constraint.PointInHalfSpace(scene, plane.getObject('dofs'), dofs)
    
    world = rigid.Body(scene, 'world', draw = True)
    world.fixed = True

    body.fixed = True
    
    # hinge = scene.createChild('hinge')
    # joint = rigid.HingeJoint(hinge, world.dofs, body.dofs)

    
    

