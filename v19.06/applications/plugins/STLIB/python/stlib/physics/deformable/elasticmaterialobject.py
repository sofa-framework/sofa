# -*- coding: utf-8 -*-
import Sofa
from splib.objectmodel import SofaPrefab, SofaObject
from stlib.scene import Node
from stlib.visuals import VisualModel


@SofaPrefab
class ElasticMaterialObject(SofaObject):
    """Creates an object composed of an elastic material."""

    def __init__(self,
                 attachedTo=None,
                 volumeMeshFileName=None,
                 name="ElasticMaterialObject",
                 rotation=[0.0, 0.0, 0.0],
                 translation=[0.0, 0.0, 0.0],
                 scale=[1.0, 1.0, 1.0],
                 surfaceMeshFileName=None,
                 collisionMesh=None,
                 withConstrain=True,
                 surfaceColor=[1.0, 1.0, 1.0],
                 poissonRatio=0.3,
                 youngModulus=18000,
                 totalMass=1.0, solver=None):

        self.node = Node(attachedTo, name)
        ElasticMaterialObject.createPrefab(self,
                                           volumeMeshFileName, name, rotation, translation, scale, surfaceMeshFileName,
                                           collisionMesh, withConstrain, surfaceColor, poissonRatio, youngModulus, totalMass, solver)

    @staticmethod
    def createPrefab(self,
                     volumeMeshFileName=None,
                     name="ElasticMaterialObject",
                     rotation=[0.0, 0.0, 0.0],
                     translation=[0.0, 0.0, 0.0],
                     scale=[1.0, 1.0, 1.0],
                     surfaceMeshFileName=None,
                     collisionMesh=None,
                     withConstrain=True,
                     surfaceColor=[1.0, 1.0, 1.0],
                     poissonRatio=0.3,
                     youngModulus=18000,
                     totalMass=1.0, solver=None):

        if not self.getRoot().getObject("SofaSparseSolver", warning=False):
            if not self.getRoot().getObject("/Config/SofaSparseSolver", warning=False):
                    Sofa.msg_info("Missing RequiredPlugin SofaSparseSolver in the scene, add it from Prefab ElasticMaterialObject.")
                    self.getRoot().createObject("RequiredPlugin", name="SofaSparseSolver")

        if self.node is None:
            Sofa.msg_error("Unable to create the elastic object because it is not attached to any node. Please fill the attachedTo parameter")
            return None

        if volumeMeshFileName is None:
            Sofa.msg_error(self.node, "Unable to create an elastic object because there is no volume mesh provided.")
            return None

        if volumeMeshFileName.endswith(".msh"):
            self.loader = self.node.createObject('MeshGmshLoader', name='loader', filename=volumeMeshFileName, rotation=rotation, translation=translation, scale3d=scale)
        elif volumeMeshFileName.endswith(".gidmsh"):
            self.loader = self.node.createObject('GIDMeshLoader', name='loader', filename=volumeMeshFileName, rotation=rotation, translation=translation, scale3d=scale)
        else:
            self.loader = self.node.createObject('MeshVTKLoader', name='loader', filename=volumeMeshFileName, rotation=rotation, translation=translation, scale3d=scale)

        if solver is None:
            self.integration = self.node.createObject('EulerImplicitSolver', name='integration')
            self.solver = self.node.createObject('SparseLDLSolver', name="solver")

        self.container = self.node.createObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
        self.dofs = self.node.createObject('MechanicalObject', template='Vec3d', name='dofs')

        # To be properly simulated and to interact with gravity or inertia forces, an elasticobject
        # also needs a mass. You can add a given mass with a uniform distribution for an elasticobject
        # by adding a UniformMass component to the elasticobject node
        self.mass = self.node.createObject('UniformMass', totalMass=totalMass, name='mass')

        # The next component to add is a FEM forcefield which defines how the elasticobject reacts
        # to a loading (i.e. which deformations are created from forces applied onto it).
        # Here, because the elasticobject is made of silicone, its mechanical behavior is assumed elastic.
        # This behavior is available via the TetrahedronFEMForceField component.
        self.forcefield = self.node.createObject('TetrahedronFEMForceField', template='Vec3d',
                                                 method='large', name='forcefield',
                                                 poissonRatio=poissonRatio,  youngModulus=youngModulus)

        if withConstrain:
            self.node.createObject('LinearSolverConstraintCorrection', solverName=self.solver.name)

        if collisionMesh:
            self.addCollisionModel(collisionMesh, rotation, translation, scale)

        if surfaceMeshFileName:
                self.addVisualModel(surfaceMeshFileName, surfaceColor, rotation, translation, scale)

    def addCollisionModel(self, collisionMesh, rotation=[0.0, 0.0, 0.0], translation=[0.0, 0.0, 0.0], scale=[1., 1., 1.]):
        self.collisionmodel = self.node.createChild('CollisionModel')
        self.collisionmodel.createObject('MeshSTLLoader', name='loader', filename=collisionMesh, rotation=rotation, translation=translation, scale3d=scale)
        self.collisionmodel.createObject('TriangleSetTopologyContainer', src='@loader', name='container')
        self.collisionmodel.createObject('MechanicalObject', template='Vec3d', name='dofs')
        self.collisionmodel.createObject('Triangle')
        self.collisionmodel.createObject('Line')
        self.collisionmodel.createObject('Point')
        self.collisionmodel.createObject('BarycentricMapping')

    def addVisualModel(self, filename, color, rotation, translation, scale=[1., 1., 1.]):
        self.visualmodel = VisualModel(parent=self.node, surfaceMeshFileName=filename, color=color, rotation=rotation, translation=translation)

        # Add a BarycentricMapping to deform the rendering model to follow the ones of the
        # mechanical model.
        self.visualmodel.mapping = self.visualmodel.node.createObject('BarycentricMapping', name='mapping')


def createScene(rootNode):
    from stlib.scene import MainHeader

    MainHeader(rootNode, gravity=" 0 0 0")
    ElasticMaterialObject(rootNode, "mesh/liver.msh", "NoVisual", translation=[3.0, 0.0, 0.0])
    ElasticMaterialObject(rootNode, "mesh/liver.msh", "WithVisual", translation=[-3, 0, 0], surfaceMeshFileName="mesh/liver.obj", surfaceColor=[1.0, 0.0, 0.0])
