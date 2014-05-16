#!/usr/bin/python
import Sofa

import Flexible.IO

import sys,os
datadir=os.path.dirname(os.path.realpath(__file__))+"/data/"

nbFrames = 10
nbGaussPoints = 1

file_Dof= datadir + "dofs.py"
file_SF= datadir + "SF.py"
file_GP= datadir + "GP.py"

# scene creation method
def createScene(rootNode):
	rootNode.createObject('RequiredPlugin', pluginName="image")
	rootNode.createObject('RequiredPlugin', pluginName="Flexible")

	rootNode.createObject('VisualStyle', displayFlags="showBehaviorModels showVisual")

    	rootNode.findData('dt').value=0.05
	rootNode.findData('gravity').value='0 -9.8 0'

	rootNode.createObject('EulerImplicit',rayleighStiffness="0",rayleighMass="0")
	rootNode.createObject('CGLinearSolver', iterations=25, tolerance=1.0e-9, threshold=1.0e-9)

	snode = rootNode.createChild('Flexible')	

	snode.createObject('MeshObjLoader', name="loader", filename="mesh/torus.obj", triangulate="1")
	snode.createObject('MeshToImageEngine', template="ImageUC", name="rasterizer", src="@loader", voxelSize="0.1", padSize="1", rotateImage="true")
	snode.createObject('ImageContainer', template="ImageUC", name="image", src="@rasterizer", drawBB="false")
	snode.createObject('ImageSampler', template="ImageUC", name="sampler", src="@image", method="1", param=str(nbFrames), fixedPosition="", printLog="false")
	snode.createObject('MergeMeshes', name="merged", nbMeshes="2", position1="@sampler.fixedPosition",  position2="@sampler.position")
	snode.createObject('MechanicalObject', template="Affine", name="dof", showObject="true", showObjectScale="0.7", src="@merged")

	sf = snode.createObject('VoronoiShapeFunction', name="SF", position="@dof.rest_position", src="@image", method="0", nbRef="4")
	snode.createObject('BoxROI', template="Vec3d", box="0 -2 0 5 2 5", position="@merged.position", name="FixedROI")
	snode.createObject('FixedConstraint', indices="@FixedROI.indices")

	bnode = snode.createChild('behavior')	
	gp = bnode.createObject('ImageGaussPointSampler', name="sampler", indices="@../SF.indices", weights="@../SF.weights", transform="@../SF.transform", method="2", order="4" ,showSamplesScale="0", targetNumber=str(nbGaussPoints),clearData="1")
	
	bnode.createObject('MechanicalObject', template="F332")
	bnode.createObject('LinearMapping', template="Affine,F332")

	Enode = bnode.createChild('E')	
	Enode.createObject('MechanicalObject',  template="E332" )
	Enode.createObject('GreenStrainMapping', template="F332,E332"  )
	Enode.createObject('HookeForceField',  template="E332", youngModulus="2000.0" ,poissonRatio="0.2", viscosity="0")

	cnode = snode.createChild('collision')	
	cnode.createObject('Mesh', name="mesh", src="@../loader")
	cnode.createObject('MechanicalObject',  template="Vec3d", name="pts")
	cnode.createObject('UniformMass', totalMass="20")
	cnode.createObject('LinearMapping', template="Affine,Vec3d")

	vnode = cnode.createChild('visual')	
	vnode.createObject('VisualModel',  color="1 8e-1 8e-1")
	vnode.createObject('IdentityMapping')

	snode.createObject('PythonScriptController', filename=__file__, classname="simu_save", variables="dof SF behavior/sampler")

	print "ctrl+SPACE  will dump the simulation model in /data"		
	return 0



#-----------------------------------------------------------------------------------------------------------------------------------------

class simu_save(Sofa.PythonScriptController):
	def createGraph(self,node):
		self.node=node
		self.dof = node.getObject(self.findData('variables').value[0][0])
		self.sf = node.getObject(self.findData('variables').value[1][0])
		self.gp = node.getObject(self.findData('variables').value[2][0])

		if self.dof==None or self.sf==None or self.gp==None:
			print "PythonScriptController: components in variables not found"
			return 0
		return 0

	def onKeyPressed(self,k):
		if self.dof==None or self.sf==None or self.gp==None:
			return 0
		if ord(k)==32:   # ctrl+SPACE		
			Flexible.IO.export_AffineFrames(self.dof, file_Dof)
			Flexible.IO.export_ImageShapeFunction(self.node, self.sf, file_SF)
			Flexible.IO.export_GaussPoints(self.gp, file_GP)
			print "Simulation state saved";
		return 0

