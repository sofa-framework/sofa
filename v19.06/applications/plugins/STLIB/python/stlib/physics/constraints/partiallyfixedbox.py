# -*- coding: utf-8 -*-

def PartiallyFixedBox(attachedTo=None,
             box=[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
             fixedAxis=[1,1,1],
             name="PartiallyFixedBox",
             drawBoxes=False,
             fixAll=False,
             doUpdate=False):
    """
    Constraint a set of degree of freedom to be at a fixed position.

    Args:
        attachedTo (Sofa.Node): Node where the constraint will be applyied

        box (vec6f): Specify min/max points of the font.

        fixedAxis (vec3bool): Specify which axis should be fixed (x,y,z)

        name (str): Set the name of the FixedBox constraint.

        drawBoxes (bool): Control whether or not we display the boxes.

        fixAll (bool): If true will apply the partial fixed to all the points.

        doUpdate (bool): If true

    Structure:

    .. sourcecode:: qml

        Node : {
            name : "fixedbox",
            BoxROI,
            PartialFixedConstraint
        }

    """

    c = attachedTo.createChild(name)
    c.createObject('BoxROI', name='BoxROI', box=box, drawBoxes=drawBoxes, doUpdate=doUpdate)
    c.createObject('PartialFixedConstraint', indices='@BoxROI.indices', fixedDirections=fixedAxis, fixAll=fixAll)
    return c

def createScene(rootNode):
    from stlib.scene import MainHeader
    from stlib.physics.deformable import ElasticMaterialObject
    from stlib.physics.constraints import PartiallyFixedBox

    MainHeader(rootNode)
    target = ElasticMaterialObject(fromVolumeMesh="mesh/liver.msh",
                                   withTotalMass=0.5,
                                   attachedTo=rootNode)

    PartiallyFixedBox(box=[-4, 0, 0, 5, 5, 4], attachedTo=target,
             drawBoxes=True, fixedAxis=[0,1,0])
