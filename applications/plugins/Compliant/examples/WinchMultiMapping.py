import Sofa
import sys,os

def createScene(root):
    root.gravity = [0,-0,0]
    root.createObject('RequiredPlugin', name = 'Compliant')
    root.createObject("VisualStyle", name="visualStyle1",  displayFlags="showAll" )
    root.createObject("DefaultPipeline",  depth="6" )
    root.createObject("BruteForceDetection" )
    root.createObject("MinProximityIntersection", alarmDistance=0.02, contactDistance=0.01)
    root.createObject("DefaultContactManager" )
    root.createObject("DefaultCollisionGroupManager"  )
    root.createObject("CompliantImplicitSolver"  )
    root.createObject("SequentialSolver"  )

    rotationNode = root.createChild("rotation")
    rotationDofs = rotationNode.createObject("MechanicalObject", template="Vec1d", position="0", velocity="1", showObject=True, showObjectScale="5")
    rotationNode.createObject("UniformMass", template="Vec1d")
    #rotationNode.createObject("FixedConstraint", index="0")

    pointNode = root.createChild("points")
    pointNode.createObject("MechanicalObject", template="Vec3d", position="0 0 0   1 0 0", showObject=True)
    pointNode.createObject("EdgeSetTopologyContainer",   position="0 0 0  1 0 0",  edges="0 1")
    pointNode.createObject("UniformMass")
    pointNode.createObject("FixedConstraint", index="1")

    distanceNode = pointNode.createChild("distances")
    distanceDofs = distanceNode.createObject("MechanicalObject", template="Vec1d")
    distanceNode.createObject("DistanceMapping")


    winchNode = distanceNode.createChild("winch")
    winchNode.createObject("MechanicalObject", template="Vec1d", showObject=True, showObjectScale=10)
    winchNode.createObject("WinchMultiMapping", input="@"+distanceDofs.getPathName()+" @"+rotationDofs.getPathName(), output="@./" )
    winchNode.createObject("UniformCompliance")
    rotationNode.addChild(winchNode)

    return