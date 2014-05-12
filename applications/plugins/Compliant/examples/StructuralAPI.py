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
    root.createObject('SequentialSolver', iterations=100)
    
    
   
    
    
    ##### SIMPLE KINETIC JOINTS
    # HINGE
    hingeNode = root.createChild('hinge')
    
    hinge_body1 = StructuralAPI.RigidBody( hingeNode, "hinge_body1" )
    hinge_body1.setManually( [-5,0,0,0,0,0,1], 1, [1,1,1] )
    hinge_body1.node.createObject('FixedConstraint')
    
    hinge_body2 = StructuralAPI.RigidBody( hingeNode, "hinge_body2" )
    hinge_body2.setManually( [-5,0,0,0,0,0,1], 1, [1,1,1] )
    hinge_body2.dofs.showObject=True
    hinge_body2.dofs.showObjectScale=1
    hinge_body2.dofs.velocity = "10 10 10 10 10 10"
    hinge_body2_collisionModel = hinge_body2.addCollisionMesh( "mesh/cube.obj" )
    
    hinge = StructuralAPI.HingeRigidJoint( 2, hingeNode, "joint", hinge_body1.node, hinge_body2.node )
    hinge.addLimits(-1,0.75)
    hinge.addSpring(100)
    
    
    
    # SLIDER
    sliderNode = root.createChild('slider')
    
    slider_body1 = StructuralAPI.RigidBody( sliderNode, "slider_body1" )
    slider_body1.setManually( [-10,0,0,0,0,0,1], 1, [1,1,1] )
    slider_body1.node.createObject('FixedConstraint')
    
    slider_body2 = StructuralAPI.RigidBody( sliderNode, "slider_body2" )
    slider_body2.setManually( [-10,0,0,0,0,0,1], 1, [1,1,1] )
    slider_body2.dofs.showObject=True
    slider_body2.dofs.showObjectScale=1
    slider_body2.dofs.velocity = "10 10 10 10 10 10"
    slider_body2_collisionModel = slider_body2.addCollisionMesh( "mesh/cube.obj" )
    
    slider = StructuralAPI.SliderRigidJoint( 1, sliderNode, "joint", slider_body1.node, slider_body2.node )
    slider.addLimits(-1,5)
    slider.addSpring(100)
    
    
    # CYLINDRICAL
    cylindricalNode = root.createChild('cylindrical')
    
    cylindrical_body1 = StructuralAPI.RigidBody( cylindricalNode, "cylindrical_body1" )
    cylindrical_body1.setManually( [-15,0,0,0,0,0,1], 1, [1,1,1] )
    cylindrical_body1.node.createObject('FixedConstraint')
    
    cylindrical_body2 = StructuralAPI.RigidBody( cylindricalNode, "cylindrical_body2" )
    cylindrical_body2.setManually( [-15,0,0,0,0,0,1], 1, [1,1,1] )
    cylindrical_body2.dofs.showObject=True
    cylindrical_body2.dofs.showObjectScale=1
    cylindrical_body2.dofs.velocity = "10 10 10 10 10 10"
    cylindrical_body2_collisionModel = cylindrical_body2.addCollisionMesh( "mesh/cube.obj" )
    
    cylindrical = StructuralAPI.CylindricalRigidJoint( 1, cylindricalNode, "joint", cylindrical_body1.node, cylindrical_body2.node )
    cylindrical.addLimits(-1,5,-1,0.75)
    cylindrical.addSpring(100,100)
    
    
    
    # BALL AND SOCKET
    ballandsocketNode = root.createChild('ballandsocket')
    
    ballandsocket_body1 = StructuralAPI.RigidBody( ballandsocketNode, "ballandsocket_body1" )
    ballandsocket_body1.setManually( [-20,0,0,0,0,0,1], 1, [1,1,1] )
    ballandsocket_body1.node.createObject('FixedConstraint')
    
    ballandsocket_body2 = StructuralAPI.RigidBody( ballandsocketNode, "ballandsocket_body2" )
    ballandsocket_body2.setManually( [-20,0,0,0,0,0,1], 1, [1,1,1] )
    ballandsocket_body2.dofs.showObject=True
    ballandsocket_body2.dofs.showObjectScale=1
    ballandsocket_body2.dofs.velocity = "10 10 10 10 10 10"
    ballandsocket_body2_collisionModel = ballandsocket_body2.addCollisionMesh( "mesh/cube.obj" )
    
    ballandsocket = StructuralAPI.BallAndSocketRigidJoint( ballandsocketNode, "joint", ballandsocket_body1.node, ballandsocket_body2.node )
    ballandsocket.addLimits( -1,1,-0.5,0.5,-0.75,0.75 )
    ballandsocket.addSpring( 100,100,100 )
    
    
    # PLANAR
    planarNode = root.createChild('planar')
    
    planar_body1 = StructuralAPI.RigidBody( planarNode, "planar_body1" )
    planar_body1.setManually( [-25,0,0,0,0,0,1], 1, [1,1,1] )
    planar_body1.node.createObject('FixedConstraint')
    
    planar_body2 = StructuralAPI.RigidBody( planarNode, "planar_body2" )
    planar_body2.setManually( [-25,0,0,0,0,0,1], 1, [1,1,1] )
    planar_body2.dofs.showObject=True
    planar_body2.dofs.showObjectScale=1
    planar_body2.dofs.velocity = "10 10 10 10 10 10"
    planar_body2_collisionModel = planar_body2.addCollisionMesh( "mesh/cube.obj" )
    
    planar = StructuralAPI.PlanarRigidJoint( 0, 1, planarNode, "joint", planar_body1.node, planar_body2.node )
    planar.addLimits(-0.5,1,-3,3)
    planar.addSpring( 100,100 )
    
            
    
    ##### MORE COMPLEX EXAMPLE
    complexNode = root.createChild('complex')
    # rigid body
    body1 = StructuralAPI.RigidBody( complexNode, "body1" )
    body1.setFromMesh( "mesh/cube.obj", 1, [5,0,0,0,0,0,1] )
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
    body2 = StructuralAPI.RigidBody( complexNode, "body2" )
    body2.setManually( [10,0,0,0,0,0,1], 1, [1,1,1] )
    body2.dofs.showObject=True
    body2.dofs.showObjectScale=1
    # collision model
    body2.node.createObject("Sphere", group="1")
    # visual model
    body2.visualModel = body2.addVisualModel( "mesh/cube.obj" )
    body2.visualModel.model.setColor(0,1,0,1)
    
    ##### JOINT
    joint1 = StructuralAPI.SliderRigidJoint( 0, complexNode, "joint1", body1_offset1.node, body2.node )
    joint1.addLimits( -10, 10 )
    joint1.addDamper( 5 )
    
    