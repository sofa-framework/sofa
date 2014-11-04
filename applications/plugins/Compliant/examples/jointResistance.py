import Sofa

from Compliant import StructuralAPI, Tools

path = Tools.path( __file__ )

import sys



    
def createScene(root):
    
    
    
    
  
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehaviorModels" )
    root.dt = 0.001
    root.gravity = [0, -9.8, 0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('CompliantAttachButtonSetting')
    
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=1)
    root.createObject('SequentialSolver', iterations=100)
    root.createObject('LDLTResponse')
    
    
    
    
    ### OBJETS
    
    
    
    base = StructuralAPI.RigidBody( root, "base" )
    base.setFromMesh( "mesh/PokeCube.obj", 1, [0,-4,0,0,0,0,1], [1,10,1] )
    cm = base.addCollisionMesh( "mesh/PokeCube.obj", [1,10,1] )
    cm.triangles.group = "1"
    cm.addVisualModel()
    base.node.createObject('FixedConstraint')
    
    basearmoffset = base.addOffset( "basearmoffset", [0,4.5,0,0,0,0,1] )
    basearmoffset.dofs.showObject=True
    basearmoffset.dofs.showObjectScale=1
    
    baseslideoffset = base.addOffset( "baseslideoffset", [0,0,0,0,0,0,1] )
    baseslideoffset.dofs.showObject=True
    baseslideoffset.dofs.showObjectScale=1
    
    
    
    
    
    arm = StructuralAPI.RigidBody( root, "arm" )
    arm.setFromMesh( "mesh/PokeCube.obj", 1, [0,5,0,0,0,0,1], [.9,10,.9] )
    cm = arm.addCollisionMesh( "mesh/PokeCube.obj", [.9,10,.9] )
    cm.triangles.group = "1"
    vm = cm.addVisualModel()
    vm.model.setColor( 1,1,0,1 )
    
    armbaseoffset = arm.addOffset( "armbaseoffset", [0,-4.5,0,0,0,0,1] )
    armbaseoffset.dofs.showObject=True
    armbaseoffset.dofs.showObjectScale=1
    
    
    
    
    sliding = StructuralAPI.RigidBody( root, "sliding" )
    sliding.setFromMesh( "mesh/PokeCube.obj", 1, [0,-4,0,0,0,0,1], [6,.7,.7] )
    cm = sliding.addCollisionMesh( "mesh/PokeCube.obj", [6,.7,.7] )
    cm.triangles.group = "1"
    vm = cm.addVisualModel()
    vm.model.setColor( 0,1,0,1 )
    
    

    
    
    ##### JOINTS
    
    
    hinge = StructuralAPI.HingeRigidJoint( 2, "hinge", basearmoffset.node, armbaseoffset.node )
    hinge.addLimits(-1,1)
    hinge.addResistance(100)
    
    
    
   
    slider = StructuralAPI.SliderRigidJoint( 0, "slider", baseslideoffset.node, sliding.node )
    slider.addResistance(10)
    slider.addLimits(-3,3)
    