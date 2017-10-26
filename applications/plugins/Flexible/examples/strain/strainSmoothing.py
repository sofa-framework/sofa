import Sofa
from SofaPython.Tools import listToStr as concat
from SofaPython.Tools import listListToStr as lconcat

def createScene(rootNode) :

	# rootNode.findData('gravity').value='0 -1 0'
	rootNode.findData('gravity').value='0 0 0'
	rootNode.findData('dt').value='10'

	rootNode.createObject('BackgroundSetting',color='1 1 1')
	rootNode.createObject('OrderIndependentTransparencyManager')
	rootNode.createObject('VisualStyle',displayFlags='showVisual')

	rootNode.createObject("RequiredPlugin", name="Compliant")
	rootNode.createObject("RequiredPlugin", name="Image")
	rootNode.createObject("RequiredPlugin", name="Flexible")

	rootNode.createObject('CompliantPseudoStaticSolver', name='odesolver', iterations="1")
	# rootNode.createObject('CompliantImplicitSolver', name='odesolver' )
	rootNode.createObject('LDLTSolver', name='numsolver')
	# rootNode.createObject('LUSolver', name='numsolver')
	# rootNode.createObject('MinresSolver', name="numsolver", iterations =25, precision=1e-14)

	# node = rootNode.createChild("beam_ref")
	# addBeam(node,[9,9,41],0,drawPlain=True)

	node = rootNode.createChild("beam_ref")
	addBeam(node,[5,5,21],0,drawPlain=True)

	# node = rootNode.createChild("beam1")
	# addBeam(node,[3,3,11],1)

	node = rootNode.createChild("beam1")
	addBeam(node,[3,3,6],1)

	node = rootNode.createChild("beam_smooth1")
	addBeam(node,[3,3,11],2,smooth=True)

	node = rootNode.createChild("beam_smooth2")
	addBeam(node,[3,3,6],3,smooth=True)



def addBeam(node,resolution,tr,smooth=False,drawPlain=False):
	node.createObject("RegularGrid", name="mesh", n=concat(resolution), min=concat([-0.2+tr,-0.2,-1]), max=concat([0.2+tr,0.2,1]))
	node.createObject("MechanicalObject", template="Vec3", name="dofs" )

	node.createObject("BoxROI", template="Vec3d", position="@mesh.position",  box=concat([-1+tr,-1,-1.1,1+tr,1,-0.99]) )
	node.createObject("FixedConstraint", indices="@[-1].indices" )

	node.createObject("BoxROI", template="Vec3d", position="@mesh.position", box=concat([-1+tr,-1,0.99,1+tr,1,1.1]),  drawBoxes="1" )
	node.createObject("ConstantForceField", template="Vec3d", points="@[-1].indices", totalForce="0 -3 0")
	# node.createObject("ConstantForceField", template="Vec3d", points="@[-1].indices", totalForce="0 -6 0")
	# node.createObject("ConstantForceField", template="Vec3d", points="@[-1].indices", totalForce="0 0 100")

	node.createObject("UniformMass", totalMass="10" )

	# node.createObject("HexahedronFEMForceField", youngModulus="1000.0" ,poissonRatio="0", method="polar", updateStiffnessMatrix="false" )

	node.createObject("BarycentricShapeFunction",  nbRef="8" )
	Fnode = node.createChild("F")
	GPsampler = Fnode.createObject("TopologyGaussPointSampler", name="sampler", inPosition="@../mesh.position", showSamplesScale="0", method="0", order="2" )
	Fnode.createObject("MechanicalObject",  template="F331", name="F" )
	mapping = Fnode.createObject("LinearMapping", name="FMapping", template="Vec3d,F331"   )

	Enode = Fnode.createChild("E")
	Enode.createObject("MechanicalObject",  template="E331", name="E"  )
	# Enode.createObject("CorotationalStrainMapping", template="F331,E331",   method="svd",  geometricStiffness=True)
	Enode.createObject("GreenStrainMapping", template="F331,E331",  geometricStiffness=True)

	if smooth:
		Esnode = Enode.createChild("Es")

		# Esnode.createObject("HatShapeFunction",  nbRef="8", param="0.5", position=GPsampler.getLinkPath()+".position" )
		Esnode.createObject("ShepardShapeFunction",  nbRef="8", position=GPsampler.getLinkPath()+".position" )
		Esnode.createObject("RegularGrid", name="sampler", n=concat([2*r for r in resolution]), min=concat([-0.2+tr,-0.2,-1]), max=concat([0.2+tr,0.2,1]))
		Esnode.createObject("GaussPointSmoother", name="GP", position="@sampler.position", inputVolume=GPsampler.getLinkPath()+".volume", inputTransforms=GPsampler.getLinkPath()+".transforms" )
		Esnode.createObject("MechanicalObject",  template="E331", name="E"  )
		Esnode.createObject("LinearStrainMapping", template="E331,E331", indices="@GP.indices", weights="@GP.weights")
		Esnode.createObject("HookeForceField",  template="E331", name="ff", youngModulus="3000.0", poissonRatio="0.3", viscosity="0"    )
	else:
		Enode.createObject("HookeForceField",  template="E331", name="ff", youngModulus="3000.0", poissonRatio="0.3", viscosity="0"    )

	vnode = node.createChild("Visual")
	if drawPlain:
		vnode.createObject("VisualModel", color="0.8 0.8 1 1")
	else:
		vnode.createObject("VisualModel", color="0.8 0.8 1 1",edges="@../mesh.edges",position="@../mesh.position")
	vnode.createObject("IdentityMapping" )
