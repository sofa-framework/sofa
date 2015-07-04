import Sofa
import Compliant

import math

from Compliant import Rigid, Tools
from Compliant.types import Quaternion

path = Tools.path( __file__ )

def createScene(node):
    scene = Tools.scene( node )

    style = node.getObject('style')
    style.findData('displayFlags').showMappings = True

    manager = node.getObject('manager')
    manager.response = 'FrictionCompliantContact'
    
    # node.createObject('CompliantAttachButton')
    
    
    globalMu = 0.7 # per object friction coefficient (the friction coef between 2 objects is approximated as the product of both coefs)
    
    manager.responseParams = 'mu={0}&compliance={1}&horizontalConeProjection=1'.format(globalMu,
                                                                                       1e-8)  
                            

    ode = node.getObject('ode')
    ode.stabilization = "pre-stabilization"
    ode.debug = 2


    # (un)comment these to see anisotropy issues with sequential solver
    solver = 'ModulusSolver'
    solver = 'SequentialSolver'
    
    num = node.createObject(solver,
                            name = 'num',
                            iterations = 100,
                            precision = 1e-14,
                            anderson = 4)
    num.printLog = True
    
    proximity = node.getObject('proximity')

    proximity.alarmDistance = 0.5
    proximity.contactDistance = 0.2
    proximity.useLineLine = True

  
    # plane
    plane = Rigid.Body('plane')
    plane.dofs.translation = [0, 5, -15]

    alpha = math.pi / 5
    mu = math.tan( alpha )

    print "plane mu:", mu, 
    if mu < globalMu: print '(should stick)'
    else: print '(should slide)'

    q = Quaternion.exp( [alpha, 0.0, 0.0] )

    plane.dofs.rotation = q
    s = 6
    plane.scale = [s, s, s]
    
    plane.visual = path + '/../mesh/ground.obj'
    plane.collision = plane.visual
    plane.mass_from_mesh( plane.visual, 10 )
    plane.mu = 0.5 # per object friction coefficient
    plane.node = plane.insert( scene )
    plane.node.createObject('FixedConstraint', indices = '0')

    # box
    box = Rigid.Body('box')
    box.visual = path + '/../mesh/cube.obj'
    box.collision = box.visual
    box.dofs.translation = [0, 3, 0]
    box.mass_from_mesh( box.visual, 50 )
    box.mu = 1 # per object friction coefficient

    box.dofs.rotation = q


    box.node = box.insert( scene )

