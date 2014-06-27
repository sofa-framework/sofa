# This scene demonstrates the ability to create a DAG in Python
# the graph hyerarchy will be of this form:
#         root
#         / \
#        A   B
#         \ /
#          C

import Sofa

# this function only creates a dragon object already setup
def createDragon(parentnode,name,color):

	node=parentnode.createChild(name);
	node.createObject('EulerImplicit',name='cg_odesolver',printLog='false')
	node.createObject('CGLinearSolver',name='linear solver',iterations=25,tolerance=1.0e-9,threshold=1.0e-9)
	node.createObject('MechanicalObject')
	node.createObject('UniformMass',totalmass=10)
	node.createObject('RegularGrid',nx=6, ny=5, nz=3, xmin=-11, xmax=11, ymin=-7, ymax=7, zmin=-4, zmax=4 )
	node.createObject('RegularGridSpringForceField', name='Springs', stiffness=350, damping=1)

	VisuNode = node.createChild('VisuDragon')
	VisuNode.createObject('OglModel',name='Visual',filename='mesh/dragon.obj',color=color)
	VisuNode.createObject('BarycentricMapping',input='@..',output='@Visual')

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MeshObjLoader', name='loader', filename='mesh/dragon.obj')
	SurfNode.createObject('Mesh',src='@loader')
	SurfNode.createObject('MechanicalObject',src='@loader')
	SurfNode.createObject('Triangle')
	SurfNode.createObject('Line')
	SurfNode.createObject('Point')
	SurfNode.createObject('BarycentricMapping')

	return node








# This standard entry point is called to create the scene, when loading directly this script from runSofa
# WARNING: you must use the DAG simulation: "runSofa -s dag"
def createScene(rootNode):
	A = rootNode.createChild('A')
	B = rootNode.createChild('B')
	C = A.createChild('C')
	B.addChild(C)

	#just for the fun...
	createDragon(C,"Smaug","red")
	


