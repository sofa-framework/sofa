import Sofa
from SofaPython.Tools import listToStr as concat
from SofaPython.Tools import listListToStr as lconcat
from SofaPython import Quaternion as quat
import numpy as np

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
	rootNode.createObject('MinresSolver', name="numsolver", iterations =25, precision=1e-14)

	addModel(rootNode.createChild("model_ref"),[21,51],0, useQuads=True, smooth=False, drawPlain=True)
	addModel(rootNode.createChild("model"),[6,11],0.1, useQuads=False, smooth=False)
	addModel(rootNode.createChild("model_smooth"),[6,11],0.2, useQuads=False, smooth=True)



def addModel(node,resolution,tr,useQuads=False, smooth=False,drawPlain=False):

	# node.createObject("MeshObjLoader", name="loader", filename="mesh/sphere_"+resolution+".obj", triangulate="1", scale="1 1 1")
	# node.createObject("Mesh", src="@loader", name="mesh")
	# node.createObject("SurfacePressureForceField", pressure="1E2", volumeConservationMode='1', drawForceScale="0.00001")

	if useQuads:
		node.createObject("GridMeshCreator", name="loader" ,filename="nofile", resolution=concat(resolution), trianglePattern="0", translation="0 0 "+str(tr), rotation="0 0 0", scale3d="0.2 1 0" )
		node.createObject("QuadSetTopologyContainer",  name="mesh", src="@loader" )
	else:
		node.createObject("GridMeshCreator", name="loader", filename="nofile", resolution=concat(resolution), trianglePattern="1", translation="0 0 "+str(tr), rotation="0 0 0", scale3d="0.2 1 0" )
		node.createObject("MeshTopology", name="mesh", src="@loader" )

	node.createObject("MechanicalObject", src="@loader" )
	node.createObject("BarycentricShapeFunction" )

	node.createObject("UniformMass",  totalMass="1000")


	g = 0.0
	node.createObject("BoxROI", name="box1", box=concat([g-0.005,-0.005,-0.005+tr,1.005-g,0.005 ,0.005+tr]) )
	node.createObject("FixedConstraint", indices="@[-1].indices" )
	node.createObject("BoxROI", name="box2", box=concat([g-0.005,0.995,-0.005+tr,1.005-g,1.005 ,0.005+tr]) )
	# node.createObject("FixedConstraint", indices="@[-1].indices" )
	node.createObject("ConstantForceField", template="Vec3d", points="@[-1].indices", totalForce="-3 -3 0")

	node.createObject("PartialFixedConstraint", fixedDirections="0 0 1", fixAll=True )

	Fnode = node.createChild("F")
	GPsampler = Fnode.createObject("TopologyGaussPointSampler", name="sampler", inPosition="@../loader.position", method="0", order="2" if useQuads else "1", orientation="0 0 25", useLocalOrientation="0")
	Fnode.createObject("MechanicalObject", template="F321"  )
	Fnode.createObject("LinearMapping", template="Vec3d,F321", showDeformationGradientScale="0.0", showDeformationGradientStyle="1")
	Enode = Fnode.createChild("E")
	Enode.createObject("MechanicalObject", template="E321", name="E")
	Enode.createObject("GreenStrainMapping", template="F321,E321" )
	# Enode.createObject("CorotationalStrainMapping", template="F321,E321",   method="polar",  geometricStiffness=True)

	if smooth:
		Esnode = Enode.createChild("Es")
		# Esnode.createObject("HatShapeFunction",  nbRef="4", param="0.5", position=GPsampler.getLinkPath()+".position" )
		Esnode.createObject("ShepardShapeFunction",  nbRef="4", position=GPsampler.getLinkPath()+".position" )
		Esnode.createObject("GridMeshCreator", name="sampler" ,filename="nofile", resolution=concat([2*r for r in resolution]), trianglePattern="0", translation="0 0 "+str(tr), rotation="0 0 0", scale3d="0.2 1 0" )
		Esnode.createObject("GaussPointSmoother", name="GP", position="@sampler.position", showSamplesScale=0, inputVolume=GPsampler.getLinkPath()+".volume", inputTransforms=GPsampler.getLinkPath()+".transforms" )
		Esnode.createObject("MechanicalObject",  template="E321", name="E"  )
		Esnode.createObject("LinearStrainMapping", template="E321,E321", indices="@GP.indices", weights="@GP.weights")
		Esnode.createObject("HookeForceField",  template="E321", name="ff", youngModulus="3000.0", poissonRatio="0.4", viscosity="0"    )
	else:
		# Enode.createObject("HookeOrthotropicForceField",  template="E321", youngModulusX="200", youngModulusY="20", poissonRatioXY="0.4", shearModulusXY="3.6", viscosity="0")
		Enode.createObject("HookeForceField",  template="E321", name="ff", youngModulus="3000.0", poissonRatio="0.4", viscosity="0"    )

	vnode = node.createChild("Visual")
	if drawPlain:
		vnode.createObject("VisualModel", color="0.8 0.8 1 1")
	else:
		vnode.createObject("VisualModel", color="0.8 0.8 1 1",edges="@../mesh.edges",position="@../mesh.position")
	vnode.createObject("IdentityMapping" )
