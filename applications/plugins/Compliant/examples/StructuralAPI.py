import Sofa

from Compliant import StructuralAPI

def createFixedRigidBody(node,name,pos):
    body = StructuralAPI.RigidBody( node, name )
    body.setManually( [pos,0,0,0,0,0,1], 1, [1,1,1] )
    body.node.createObject('FixedConstraint')
    return body
  
def createRigidBody(node,name,posx,posy=0):
    body = StructuralAPI.RigidBody( node, name )
    body.setManually( [posx,posy,0,0,0,0,1], 1, [1,1,1] )
    # enforce rigid visualization
    body.dofs.showObject=True
    body.dofs.showObjectScale=1
    # add initial velocity to make it move!
    body.dofs.velocity = "10 10 10 10 10 10"
    # add a collision model to be able to rotate it with the mouse
    collisionModel = body.addCollisionMesh( "mesh/cube.obj" )
    return body
    
    
    
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
    root.createObject('LDLTResponse')
    
    
   
    
    
    ##### SIMPLE KINETIC JOINTS
    # HINGE
    hingeNode = root.createChild('hinge')
    hinge_body1 = createFixedRigidBody(hingeNode, "hinge_body1", -25 )
    hinge_body2 = createRigidBody(hingeNode, "hinge_body2", -25 )
    hinge = StructuralAPI.HingeRigidJoint( 2, hingeNode, "joint", hinge_body1.node, hinge_body2.node )
    hinge.addLimits(-1,0.75)
    hinge.addSpring(100)
    
    
    
    # SLIDER
    sliderNode = root.createChild('slider')
    slider_body1 = createFixedRigidBody(sliderNode, "slider_body1", -20 )
    slider_body2 = createRigidBody(sliderNode, "slider_body2", -20 )
    slider = StructuralAPI.SliderRigidJoint( 1, sliderNode, "joint", slider_body1.node, slider_body2.node )
    slider.addLimits(-1,5)
    slider.addSpring(100)
    
    
    # CYLINDRICAL
    cylindricalNode = root.createChild('cylindrical')
    cylindrical_body1 = createFixedRigidBody(cylindricalNode, "cylindrical_body1", -15 )
    cylindrical_body2 = createRigidBody(cylindricalNode, "cylindrical_body2", -15 )
    cylindrical = StructuralAPI.CylindricalRigidJoint( 1, cylindricalNode, "joint", cylindrical_body1.node, cylindrical_body2.node )
    cylindrical.addLimits(-1,5,-1,0.75)
    cylindrical.addSpring(100,100)
    
    
    
    # BALL AND SOCKET
    ballandsocketNode = root.createChild('ballandsocket')
    ballandsocket_body1 = createFixedRigidBody(ballandsocketNode, "ballandsocket_body1", -10 )
    ballandsocket_body2 = createRigidBody(ballandsocketNode, "ballandsocket_body2", -10 )
    ballandsocket = StructuralAPI.BallAndSocketRigidJoint( ballandsocketNode, "joint", ballandsocket_body1.node, ballandsocket_body2.node )
    ballandsocket.addLimits( -1,1,-0.5,0.5,-0.75,0.75 )
    ballandsocket.addSpring( 100,100,100 )
    
    
    # PLANAR
    planarNode = root.createChild('planar')
    planar_body1 = createFixedRigidBody(planarNode, "planar_body1", -5 )
    planar_body2 = createRigidBody(planarNode, "planar_body2", -5 )
    planar = StructuralAPI.PlanarRigidJoint( 2, planarNode, "joint", planar_body1.node, planar_body2.node )
    planar.addLimits(-0.5,1,-3,3)
    planar.addSpring( 100,100 )
    
            
    # GIMBAL
    gimbalNode = root.createChild('gimbal')
    gimbal_body1 = createFixedRigidBody(gimbalNode, "gimbal_body1", 0 )
    gimbal_body2 = createRigidBody(gimbalNode, "gimbal_body2", 0 )
    gimbal = StructuralAPI.GimbalRigidJoint( 2, gimbalNode, "joint", gimbal_body1.node, gimbal_body2.node )
    gimbal.addLimits(-0.5,1,-3,3)
    gimbal.addSpring( 100,100 )
    
    # FIXED
    fixedNode = root.createChild('fixed')
    fixed_body1 = createFixedRigidBody(hingeNode, "fixed_body1", 5 )
    fixed_body2 = createRigidBody(hingeNode, "fixed_body2", 5 )
    fixed = StructuralAPI.FixedRigidJoint( fixedNode, "joint", fixed_body1.node, fixed_body2.node )
  
    # DISTANCE
    distanceNode = root.createChild('distance')
    distance_body1 = createFixedRigidBody(distanceNode, "distance_body1", 10 )
    distance_body2 = createRigidBody(distanceNode, "distance_body2", 10, 4 )
    distance = StructuralAPI.DistanceRigidJoint( distanceNode, "joint", distance_body1.node, distance_body2.node )
  
  
  
  
            
    ##### MORE COMPLEX EXAMPLE
    complexNode = root.createChild('complex')
    # rigid body
    body1 = StructuralAPI.RigidBody( complexNode, "body1" )
    body1.setFromMesh( "mesh/cube.obj", 1, [5,-7,0,0,0,0,1] )
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
    body2.setManually( [10,-7,0,0,0,0,1], 1, [1,1,1] )
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
    
    