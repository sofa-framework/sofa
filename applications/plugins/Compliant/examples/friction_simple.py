import Sofa
import Plugin

import math

from Compliant import Rigid, Tools

dir = Tools.path( __file__ )

def createScene(node):
    scene = Tools.scene( node )

    style = node.getObject('style')
    style.findData('displayFlags').showMappings = True

    manager = node.getObject('manager')
    manager.response = 'CompliantContact'
    # manager.responseParams = 'mu=0.5' 

    ode = node.getObject('ode')
    ode.stabilization = False
    ode.debug = True

    num = node.createObject('QPSolver',
                            name = 'num',
                            iterations = 100,
                            precision = 1e-14)
    
    proximity = node.getObject('proximity')

    proximity.alarmDistance = 0.5
    proximity.contactDistance = 0.2

  
    # plane
    plane = Rigid.Body('plane')
    plane.visual = dir + '/mesh/ground.obj'
    plane.collision = plane.visual
    plane.mass_from_mesh( plane.visual, 10 )
    plane.node = plane.insert( scene )
    plane.node.createObject('FixedConstraint', indices = '0')
    
    # box
    box = Rigid.Body('box')
    box.visual = dir + '/mesh/cube.obj'
    box.collision = box.visual
    box.dofs.translation = [0, 3, 0]
    box.mass_from_mesh( box.visual, 50 )
    box.node = box.insert( scene )

   
