import Sofa

from Compliant import StructuralAPI


def createScene(root):
  
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehavior showWireframe showCollisionModels" )
    root.dt = 0.001
    root.gravity = [0, -9.8, 0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('CompliantAttachButtonSetting')
    
    ##### SOLVER
    root.createObject('AssembledSolver', stabilization=1)
    root.createObject('MinresSolver', iterations=100)
    
    ##### RIGID BODY 1
    # rigid body
    body1 = StructuralAPI.RigidBody( root, "body1" )
    body1.setFromMesh( "mesh/cube.obj", 1 )
    body1.dofs.showObject=True
    body1.dofs.showObjectScale=1
    # collision model
    body1_collisionModel = body1.addCollisionMesh( "mesh/cube.obj", [3.1,1.1,2.1] )
    body1_collisionModel.triangles.group = "1"
    # visual model
    body1.visualModel = body1.addVisualModel( "mesh/cube.obj", [3,1,2] )
    body1.visualModel.model.setColor(1,0,0,1)
    # offsets
    body1_offset1 = body1.addOffset( "offset1", [2,1,3,0.7325378163287418,0.4619397662556433,-0.19134171618254486,0.4619397662556433] )
    body1_offset1.dofs.showObject=True
    body1_offset1.dofs.showObjectScale=.5
    body1_offset2 = body1.addOffset( "offset2", [1,0,0,0,0,0,1] )
    body1_offset2.dofs.showObject=True
    body1_offset2.dofs.showObjectScale=.5
    
    ##### RIGID BODY 2
    # rigid body
    body2 = StructuralAPI.RigidBody( root, "body2" )
    body2.setManually( [10,0,0,0,0,0,1], 1, [1,1,1] )
    body2.dofs.showObject=True
    body2.dofs.showObjectScale=1
    # collision model
    body2.node.createObject("Sphere", group="1")
    # visual model
    body2.visualModel = body2.addVisualModel( "mesh/cube.obj" )
    body2.visualModel.model.setColor(0,1,0,1)
    
    ##### JOINT
    joint1 = StructuralAPI.GenericRigidJoint( root, "joint1", body1_offset1.node, body2.node, [0,1,1,1,1,1] )
    
