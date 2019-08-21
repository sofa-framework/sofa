# -*- coding: utf-8 -*-
"""
Templates to rigidify a deformable object.

The rigidification consist in mixing in a single object rigid and deformable parts.
The rigid and deformable parts are interacting together.

**Sofa Templates:**

.. autosummary::

    Rigidify

stlib.physics.mixedmaterial.Rigidify
*******************************
.. autofunction:: Rigidify


Contributors:
        damien.marchal@univ-lille.fr
        eulalie.coevoet@inria.fr
"""

import Sofa
from splib.numerics import Vec3, Quat, sdiv


def getBarycenter(selectedPoints):
    poscenter = [0., 0., 0.]
    if len(selectedPoints) != 0:
            poscenter = sdiv(sum(selectedPoints), float(len(selectedPoints)))
    return poscenter


def Rigidify(targetObject, sourceObject, groupIndices, frames=None, name=None, frameOrientation=None):
        """ Transform a deformable object into a mixed one containing both rigid and deformable parts.

            :param targetObject: parent node where to attach the final object.
            :param sourceObject: node containing the deformable object. The object should be following
                                 the ElasticMaterialObject template.
            :param list groupIndices: array of array indices to rigidify. The length of the array should be equal to the number
                                      of rigid component.
            :param list frames: array of frames. The length of the array should be equal to the number
                                of rigid component. The orientation are given in eulerAngles (in degree) by passing
                                three values or using a quaternion by passing four values.
                                [[rx,ry,rz], [qx,qy,qz,w]]
                                User can also specify the position of the frame by passing six values (position and orientation in degree)
                                or seven values (position and quaternion).
                                [[x,y,z,rx,ry,rz], [x,y,z,qx,qy,qz,w]]
                                If the position is not specified, the position of the rigids will be the barycenter of the region to rigidify.
            :param str name: specify the name of the Rigidified object, is none provided use the name of the SOurceObject.
        """
        # Deprecation Warning
        if frameOrientation is not None:
            Sofa.msg_warning("The parameter frameOrientations of the function Rigidify is now deprecated. Please use frames instead.")
            frames = frameOrientation

        if frames is None:
            frames = [[0., 0., 0.]]*len(groupIndices)

        assert len(groupIndices) == len(frames), "size mismatch."

        if name is None:
                name = sourceObject.name

        sourceObject.init()
        ero = targetObject.createChild(name)

        allPositions = sourceObject.container.position
        allIndices = map(lambda x: x[0], sourceObject.container.points)

        rigids = []
        indicesMap = []

        def mfilter(si, ai, pts):
                tmp = []
                for i in ai:
                        if i in si:
                                tmp.append(pts[i])
                return tmp

        # get all the points from the source.
        sourcePoints = map(Vec3, sourceObject.dofs.position)
        selectedIndices = []
        for i in range(len(groupIndices)):
                selectedPoints = mfilter(groupIndices[i], allIndices, sourcePoints)

                if len(frames[i]) == 3:
                        orientation = Quat.createFromEuler(frames[i], inDegree=True)
                        poscenter = getBarycenter(selectedPoints)
                elif len(frames[i]) == 4:
                        orientation = frames[i]
                        poscenter = getBarycenter(selectedPoints)
                elif len(frames[i]) == 6:
                        orientation = Quat.createFromEuler([frames[i][3], frames[i][4], frames[i][5]], inDegree=True)
                        poscenter = [frames[i][0], frames[i][1], frames[i][2]]
                elif len(frames[i]) == 7:
                        orientation = [frames[i][3], frames[i][4], frames[i][5], frames[i][6]]
                        poscenter = [frames[i][0], frames[i][1], frames[i][2]]
                else:
                        Sofa.msg_error("Do not understand the size of a frame.")

                rigids.append(poscenter + list(orientation))

                selectedIndices += map(lambda x: x, groupIndices[i])
                indicesMap += [i] * len(groupIndices[i])

        otherIndices = filter(lambda x: x not in selectedIndices, allIndices)
        Kd = {v: None for k, v in enumerate(allIndices)}
        Kd.update({v: [0, k] for k, v in enumerate(otherIndices)})
        Kd.update({v: [1, k] for k, v in enumerate(selectedIndices)})
        indexPairs = [v for kv in Kd.values() for v in kv]

        freeParticules = ero.createChild("DeformableParts")
        freeParticules.createObject("MechanicalObject", template="Vec3", name="dofs",
                                    position=[allPositions[i] for i in otherIndices])

        rigidParts = ero.createChild("RigidParts")
        rigidParts.createObject("MechanicalObject", template="Rigid3", name="dofs", reserve=len(rigids), position=rigids)

        rigidifiedParticules = rigidParts.createChild("RigidifiedParticules")
        rigidifiedParticules.createObject("MechanicalObject", template="Vec3", name="dofs",
                                          position=[allPositions[i] for i in selectedIndices])
        rigidifiedParticules.createObject("RigidMapping", name="mapping", globalToLocalCoords='true', rigidIndexPerPoint=indicesMap)

        sourceObject.removeObject(sourceObject.solver)
        sourceObject.removeObject(sourceObject.integration)
        sourceObject.removeObject(sourceObject.LinearSolverConstraintCorrection)

        # The coupling is made with the sourceObject. If the source object is from an ElasticMaterialObject
        # We need to get the owning node form the current python object (this is a hack because of the not yet
        # Finalized design of stlib.
        coupling = sourceObject
        if hasattr(sourceObject, "node"):
            coupling = sourceObject.node

        coupling.createObject("SubsetMultiMapping", name="mapping", template="Vec3,Vec3",
                              input=freeParticules.dofs.getLinkPath()+" "+rigidifiedParticules.dofs.getLinkPath(),
                              output=sourceObject.dofs.getLinkPath(),
                              indexPairs=indexPairs)

        rigidifiedParticules.addChild(coupling)
        freeParticules.addChild(coupling)
        return ero


def createScene(rootNode):
        """
        """
        from stlib.scene import MainHeader
        from stlib.physics.deformable import ElasticMaterialObject
        from stlib.physics.mixedmaterial import Rigidify
        from splib.objectmodel import setData

        MainHeader(rootNode, plugins=["SofaSparseSolver"])
        rootNode.VisualStyle.displayFlags = "showBehavior"

        modelNode = rootNode.createChild("Modeling")
        elasticobject = ElasticMaterialObject(modelNode, "mesh/liver.msh", "ElasticMaterialObject")

        # Rigidification of the elasticobject for given indices with given frameOrientations.
        o = Rigidify(modelNode,
                     elasticobject,
                     name="RigidifiedStructure",
                     frames=[[0., 0., 0], [0., 0., 0]],
                     groupIndices=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [48, 49, 50, 51]])

        # Activate some rendering on the rigidified object.
        setData(o.RigidParts.dofs, showObject=True, showObjectScale=1, drawMode=2)
        setData(o.RigidParts.RigidifiedParticules.dofs, showObject=True, showObjectScale=0.1,
                drawMode=1, showColor=[1., 1., 0., 1.])
        setData(o.DeformableParts.dofs, showObject=True, showObjectScale=0.1, drawMode=2)
        o.RigidParts.createObject("FixedConstraint", indices=0)

        simulationNode = rootNode.createChild("Simulation")
        simulationNode.createObject("EulerImplicitSolver")
        simulationNode.createObject("CGLinearSolver")
        simulationNode.addChild(o)
        return rootNode
