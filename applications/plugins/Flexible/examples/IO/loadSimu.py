#!/usr/bin/python
import Sofa

import sys,os
datadir=os.path.dirname(os.path.realpath(__file__))+"/data/"
sys.path.insert(0, datadir)

file_Dof= "dofs"
file_SF= "SF"
file_GP= "GP"

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

	__import__(file_Dof).loadDofs(snode)
	
	snode.createObject('BoxROI', template="Vec3d", box="0 -2 0 5 2 5", position="@dofs.rest_position", name="FixedROI")
	snode.createObject('FixedConstraint', indices="@FixedROI.indices")

	__import__(file_SF).loadSF(snode)

	bnode = snode.createChild('behavior')	

	bnode.createObject('ImageGaussPointSampler', name="sampler", indices="@../SF.indices", weights="@../SF.weights", transform="@../SF.transform", method="2", order="4" ,showSamplesScale="0", targetNumber="1")
#	__import__(file_GP).loadGPs(bnode)

	bnode.createObject('MechanicalObject', template="F332")
	bnode.createObject('LinearMapping', template="Affine,F332")
	Enode = bnode.createChild('E')	
	Enode.createObject('MechanicalObject',  template="E332" )
	Enode.createObject('GreenStrainMapping', template="F332,E332"  )
	Enode.createObject('HookeForceField',  template="E332", youngModulus="2000.0" ,poissonRatio="0.2", viscosity="0")

	cnode = snode.createChild('collision')	
	cnode.createObject('MeshObjLoader', name="loader", filename="mesh/torus.obj", triangulate="1")
	cnode.createObject('Mesh', name="mesh", src="@loader")
	cnode.createObject('MechanicalObject',  template="Vec3d", name="pts")
	cnode.createObject('UniformMass', totalMass="20")
	cnode.createObject('LinearMapping', template="Affine,Vec3d")

	vnode = cnode.createChild('visual')	
	vnode.createObject('VisualModel',  color="1 8e-1 8e-1")
	vnode.createObject('IdentityMapping')

	return 0


