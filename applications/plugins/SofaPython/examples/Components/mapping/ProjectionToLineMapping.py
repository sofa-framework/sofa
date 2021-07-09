import Sofa


def createScene(node):

    node.gravity="0 0 0"

    node.createObject('VisualStyle', displayFlags="showBehaviorModels showMechanicalMappings" )
    node.createObject('EulerImplicit',vdamping=1)
    node.createObject('CGLinearSolver', iterations="25", tolerance="1e-5", threshold="1e-5")

    linenode = node.createChild("line")
    linenode.createObject('MechanicalObject', template="Vec3d", position="0 0 0   1 0 0", tags="NoPicking")
    linenode.createObject('UniformMass', template="Vec3d")


    pointnode = node.createChild("points")
    pointnode.createObject('MechanicalObject', template="Vec3d", position="0 1 0    .5 .5 .5    1 1 1", showObject="1", showObjectScale=".1", drawMode="1" )
    pointnode.createObject('UniformMass', template="Vec3d")

    projectednode = pointnode.createChild("projected")
    projectednode.createObject('MechanicalObject', template="Vec3d",showObject="1", showObjectScale=".12", drawMode="2")
    projectednode.createObject('ProjectionToLineMultiMapping', template="Vec3d,Vec3d", indices="0 1", input="@.. @/line/.",output="@.")
    linenode.addChild(projectednode)
