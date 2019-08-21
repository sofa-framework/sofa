# -*- coding: utf-8 -*-

def FixedBox(applyTo=None, atPositions=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
             name="FixedBox",
             doVisualization=False,
             position=None,
             constraintStrength='1e12', doRecomputeDuringSimulation=False):
    """
    Constraint a set of degree of freedom to be at a fixed position.

    Args:
        applyTo (Sofa.Node): Node where the constraint will be applyied

        atPosition (vec6f): Specify min/max points of the font.

        name (str): Set the name of the FixedBox constraint.

        doVisualization (bool): Control whether or not we display the boxes.

    Structure:

    .. sourcecode:: qml

        Node : {
            name : "fixedbox",
            BoxROI,
            RestShapeSpringsFroceField
        }

    """

    c = applyTo.createChild(name)
    if position == None:
        c.createObject('BoxROI', name='BoxROI', box=atPositions, drawBoxes=doVisualization, doUpdate=doRecomputeDuringSimulation)
    else:
        c.createObject('BoxROI', position=position, name='BoxROI', box=atPositions, drawBoxes=doVisualization, doUpdate=doRecomputeDuringSimulation)

    c.createObject('RestShapeSpringsForceField', points='@BoxROI.indices', stiffness='1e12')
    return c

def createScene(rootNode):
    from stlib.scene import MainHeader
    from stlib.physics.deformable import ElasticMaterialObject
    from stlib.physics.constraints import FixedBox

    MainHeader(rootNode)
    target = ElasticMaterialObject(volumeMeshFileName="mesh/liver.msh",
                                   totalMass=0.5,
                                   attachedTo=rootNode)

    FixedBox(atPositions=[-4, 0, 0, 5, 5, 4], applyTo=target,
             doVisualization=True)
