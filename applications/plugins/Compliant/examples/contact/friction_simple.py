import Sofa

import math

from Compliant import StructuralAPI, Tools

dir = Tools.path( __file__ )

def createScene(node):
    scene = Tools.scene( node )

    style = node.getObject('style')
    style.findData('displayFlags').showMappings = True

    manager = node.getObject('manager')
    manager.response = 'FrictionCompliantContact'
    
    node.createObject('CompliantAttachButton')
    
    
    globalMu = 0 # per object friction coefficient (the friction coef between 2 objects is approximated as the product of both coefs)
    manager.responseParams = 'mu='+str(globalMu)+"&horizontalConeProjection=1"  # perfom an horizontal Coulomb cone projection
                            

    ode = node.getObject('ode')
    ode.stabilization = "pre-stabilization"
    ode.debug = False

    num = node.createObject('SequentialSolver',
                            name = 'num',
                            iterations = 100,
                            precision = 1e-14)
    node.createObject( "LDLTResponse" )
    
    proximity = node.getObject('proximity')

    proximity.alarmDistance = 0.5
    proximity.contactDistance = 0.2
    proximity.useLineLine = True

  
    # plane
    mesh = dir + '/../mesh/ground.obj'
    plane = StructuralAPI.RigidBody( node, "plane" )
    plane.setManually( [0,0,0,0,0,0,1], 1, [1,1,1] )
    #body3.setFromMesh( mesh, 1 )
    cm = plane.addCollisionMesh( mesh )
    cm.addVisualModel()
    plane.node.createObject('FixedConstraint', indices = '0')
    cm.triangles.contactFriction = 0.5 # per object friction coefficient
    
    
    # box
    mesh = dir + '/../mesh/cube.obj'
    box = StructuralAPI.RigidBody( node, "box" )
    box.setFromMesh( mesh, 50, [0, 3, 0, 0,0,0,1] )
    cm = box.addCollisionMesh( mesh )
    cm.addVisualModel()
    cm.triangles.contactFriction = 1 # per object friction coefficient
    
    #box = Rigid.Body('box')
    #box.visual = dir + '/../mesh/cube.obj'
    #box.collision = box.visual
    #box.dofs.translation = [0, 3, 0]
    #box.mass_from_mesh( box.visual, 50 )
    #box.mu = 1 # per object friction coefficient
    #box.node = box.insert( scene )

   
