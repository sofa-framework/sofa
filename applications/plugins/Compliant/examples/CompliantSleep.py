import Sofa

from Compliant import StructuralAPI, Tools

path = Tools.path( __file__ )

def createFixedRigidBody(node,name,sleep,pos=[0,0,0],size=[1,1,1]):
    body = StructuralAPI.RigidBody( node, name )
    body.setManually( [pos[0],pos[1],pos[2],0,0,0,1], 1, [1,1,1] )
    body.dofs.showObject=True
    body.dofs.showObjectScale=1
    body.node.createObject('FixedConstraint')
    collisionModel = body.addCollisionMesh( "mesh/cube.obj", size )
    collisionModel.triangles.moving = False
    body.node.canChangeSleepingState=sleep
    return body

def createRigidBody(node,name,sleep,pos=[0,0,0],size=[1,1,1]):
    body = StructuralAPI.RigidBody( node, name )
    body.setManually( [pos[0],pos[1],pos[2],0,0,0,1], 1, [1,1,1] )
    body.dofs.showObject=True
    body.dofs.showObjectScale=1
    body.node.canChangeSleepingState=sleep
    collisionModel = body.addCollisionMesh( "mesh/cube.obj", size )
    return body



def createScene(root):

    ##### global parameters
    root.createObject('VisualStyle', displayFlags="hideVisual showBehavior showCollisionModels hideBoundingCollisionModels hideMapping hideOptions" )
    root.dt = 0.01
    root.gravity = [0, -9.8, 0]

    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('CompliantAttachButtonSetting')

    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=1)
    root.createObject('SequentialSolver', iterations=100)
    root.createObject('LDLTResponse')
    
    ##### Collisions
    root.createObject('DefaultCollisionGroupManager')
    root.createObject('NewProximityIntersection', alarmDistance=0.5, contactDistance=0.05)
    root.createObject('BruteForceDetection')
    root.createObject('DefaultContactManager', response='FrictionCompliantContact', responseParams="mu=0.01")
    root.createObject('DefaultPipeline', depth=6)

    ##### Sleep
    root.createObject('CompliantSleepController', printLog=1, listening=1, minTimeSinceWakeUp=1.0, immobileThreshold=0.02, rotationThreshold=0.1)
    
    ##### Bodies
    ground_body = createFixedRigidBody(root, 'ground_body', True, [0,0,2], [20,1,10])

    standalone_body = createRigidBody(root, 'standalone_body', True, [-6,3,0])
    standalone_body.dofs.velocity = "2 0 0 0 0 0"

    distanceNode = root.createChild('distance')
    distance_body1 = createRigidBody(distanceNode, 'distance_body1', True, [0,3,0] )
    distance_body2 = createRigidBody(distanceNode, 'distance_body2', True, [0,-3,0] )
    distance = StructuralAPI.DistanceRigidJoint( 'joint', distance_body1.node, distance_body2.node )

