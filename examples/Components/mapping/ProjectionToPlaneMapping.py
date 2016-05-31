import Sofa


def createScene(node):

    node.gravity="0 0 0"

    node.createObject('VisualStyle', displayFlags="showBehaviorModels showMechanicalMappings" )
    node.createObject('EulerImplicit',vdamping=1)
    node.createObject('CGLinearSolver', iterations="25", tolerance="1e-5", threshold="1e-5")

    planenode = node.createChild("plane")
    planenode.createObject('MechanicalObject', template="Vec3d", position="0 0 0   0 1 0", tags="NoPicking")
    planenode.createObject('UniformMass', template="Vec3d")


    pointnode = node.createChild("points")
    pointnode.createObject('MechanicalObject', template="Vec3d", position="0 1 0    .5 .5 .5    1 1 1", showObject="1", showObjectScale=".1", drawMode="1" )
    pointnode.createObject('UniformMass', template="Vec3d")

    projectednode = pointnode.createChild("projected")
    projectednode.createObject('MechanicalObject', template="Vec3d",showObject="1", showObjectScale=".12", drawMode="2")
    projectednode.createObject('ProjectionToPlaneMultiMapping', template="Vec3d,Vec3d", indices="0 1", input="@.. @/plane/.",output="@.")
    planenode.addChild(projectednode)


    symnode = pointnode.createChild("symmetry")
    symnode.createObject('MechanicalObject', template="Vec3d",showObject="1", showObjectScale=".12", drawMode="3")
    symnode.createObject('ProjectionToPlaneMultiMapping', template="Vec3d,Vec3d", indices="0 1", factor=2, input="@.. @/plane/.",output="@.")
    planenode.addChild(symnode)