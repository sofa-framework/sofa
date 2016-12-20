import Sofa
from SofaPython.Tools import listToStr as concat
from SofaPython.Tools import listListToStr as lconcat


def createSceneAndController(rootNode):

	rootNode.createObject('RequiredPlugin', pluginName="Flexible")
	rootNode.createObject('RequiredPlugin', pluginName="Compliant")
	rootNode.createObject('RequiredPlugin', pluginName="ContactMapping")

	rootNode.createObject('BackgroundSetting', color='1 1 1')
	rootNode.createObject('VisualStyle', displayFlags='showBehavior')

	rootNode.dt=0.01
	rootNode.gravity=[0,-10,0]

	rootNode.createObject('CompliantImplicitSolver', name="odesolver")
#    rootNode.createObject('SequentialSolver', name="numsolver", iterations =10, precision=1e-14)
	rootNode.createObject('LDLTSolver', name="numsolver")
	# rootNode.createObject('MinresSolver', name="numsolver", iterations =10, precision=1e-14)
	# rootNode.createObject('EulerImplicitSolver')
	# rootNode.createObject('CGSolver',  tolerance="1.0e-9" ,threshold="1.0e-9" )

	pos=[[0,0,0],[0,1,0],[0.1,1,0],[0.1,0.5,0],[0.2,0.5,0],[0.2,1,0],[0.3,1,0],[0.3,0,0]]
	edges = [ [i,i+1]  for i in xrange(len(pos)-1)]
	rootNode.createObject('Mesh', name="mesh", position=lconcat(pos) ,lines=lconcat(edges) )
	rootNode.createObject('MechanicalObject', template="Vec3d", name="DOFs", src="@mesh" )
	rootNode.createObject('UniformMass',  name="mass" ,totalMass="2")
	rootNode.createObject('FixedConstraint', name="FixedConstraint", indices=concat([1,2,3,4,5,6]) )

	# rootNode.createObject('UniformVelocityDampingForceField',dampingCoefficient="0.1")

	rootNode.createObject('RestShapeSpringsForceField',points="0 7", stiffness="10")

	Lnode = rootNode.createChild("L")
	Lnode.createObject('MechanicalObject', template="Vec1" )
	Lnode.createObject('LengthMapping', template="Vec3,Vec1", edges="@../mesh.lines",offset="-3.3" , geometricStiffness=1 )
	Lnode.createObject('UniformCompliance', compliance=1E-2,rayleighStiffness=1,isCompliance='0')

	Vnode = rootNode.createChild("visual")
	Vnode.createObject('VisualModel',edges="@../mesh.lines")
	Vnode.createObject('IdentityMapping')
