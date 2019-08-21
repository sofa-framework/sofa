 
import Sofa
import sys


class Cloth:


    def __init__(self, node, name, x ):

        clothNode = node.createChild(name)

        self.odesolver = clothNode.createObject('CompliantImplicitSolver', name='odesolver',neglecting_compliance_forces_in_geometric_stiffness="0")
        clothNode.createObject('LDLTSolver',schur=False)
        clothNode.createObject('LDLTResponse')

        clothNode.createObject('GridMeshCreator', name="loader", filename="nofile", resolution="20 20", trianglePattern="0", translation=str(x)+" 0 0", rotation="90 0 0 ", scale="10 10 0" )
        clothNode.createObject('MeshTopology', name="mesh", src="@loader" )
        clothNode.createObject('MechanicalObject', name="defoDOF", template="Vec3d",  src="@loader", showObject="1", showObjectScale="1")
        clothNode.createObject('BoxROI', name="box", box="-0.005 -0.005 -0.005    "+str(x+11)+" 0.005 0.005  " )
        clothNode.createObject('FixedConstraint', indices="@box.indices" )
        clothNode.createObject('UniformMass',  name="mass", mass="10000" )


        self.extensionNode = clothNode.createChild("extensionNode")
        self.extensionNode.createObject('MechanicalObject', template="Vec1d",  name="extensionsDOF" )
        self.extensionNode.createObject('EdgeSetTopologyContainer', edges="@../mesh.edges" )
        self.distancemapping = self.extensionNode.createObject('DistanceMapping', geometricStiffness="2" )
        self.extensionNode.createObject('UniformCompliance', name="ucomp", template="Vec1d", compliance="1e-15", isCompliance="1" )




def createScene(node):

    node.createObject('VisualStyle',displayFlags="showBehaviorModels showMechanicalMappings" )

    node.createObject("RequiredPlugin",name="Compliant")
    node.createObject("CompliantAttachButton",compliance="1e-12",isCompliance=True)

    clothNoStab = Cloth(node,"no stabilization",0)
    clothNoStab.odesolver.stabilization="no stabilization"
    clothNoStab.extensionNode.createObject("HolonomicConstraintValue")


    clothElasticity = Cloth(node,"Compliant constraint",11)
    clothElasticity.odesolver.stabilization="no stabilization"


    clothPreS = Cloth(node,"pre-stab",22)
    clothPreS.odesolver.stabilization="pre-stabilization"
    clothPreS.extensionNode.createObject("Stabilization")

    clothPostS = Cloth(node,"post-stab",33)
    clothPostS.odesolver.stabilization="post-stabilization assembly"
    clothPostS.extensionNode.createObject("Stabilization")


    clothB = Cloth(node,"Baumgarte stab",44)
    clothB.odesolver.stabilization="no stabilization"
    clothB.extensionNode.createObject("BaumgarteStabilization",alpha="0.5")


    sys.stdout.flush()