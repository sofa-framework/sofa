import Sofa
import random

# utility methods

def createArmadillo(parentNode,name,x,y,z,color):

	node = parentNode.createChild(name)
	node.createObject('EulerImplicit',name='cg_odesolver',printLog='false')
	node.createObject('CGLinearSolver',name='linear solver',iterations='25',tolerance='1.0e-9',threshold='1.0e-9')
	object = node.createObject('MechanicalObject',name='mObject',dx=x,dy=y,dz=z)
	mass = node.createObject('UniformMass',name='mass',totalmass='10')
	node.createObject('SparseGridTopology', n='4 4 4', fileTopology='mesh/Armadillo_verysimplified.obj')
	node.createObject('HexahedronFEMForceField', youngModulus='100')
	
	VisuNode = node.createChild('Visu')
	VisuNode.createObject('OglModel',name='Visual',filename='mesh/Armadillo_simplified.obj', color=color)
	VisuNode.createObject('BarycentricMapping',input='@..', output='@Visual' )

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MeshObjLoader', name='loader', filename='mesh/Armadillo_verysimplified.obj')
	SurfNode.createObject('Mesh',src='@loader')
	SurfNode.createObject('MechanicalObject', name='meca', src='@loader')
	SurfNode.createObject('Triangle')
	SurfNode.createObject('Line')
	SurfNode.createObject('Point')
	SurfNode.createObject('BarycentricMapping')

	return node

def randomColor():
	colorRandom = random.randint(1,6)
	col = 'white'
	if colorRandom==1:
		col = 'red'
	if colorRandom==2:
		col = 'green'
	if colorRandom==3:
		col = 'blue'
	if colorRandom==4:
		col = 'yellow'
	if colorRandom==5:
		col = 'cyan'
	if colorRandom==6:
		col = 'magenta'
	return col



# scene creation method
def createScene(rootNode):

	# scene global stuff
	rootNode.createObject('VisualStyle', displayFlags='hideBehaviorModels hideCollisionModels hideMappings hideForceFields')
	rootNode.createObject('CollisionPipeline', verbose=0, depth=10, draw=0)
	rootNode.createObject('BruteForceDetection', name='N2')
	rootNode.createObject('MinProximityIntersection', name='Proximity', alarmDistance=0.5, contactDistance=0.33)
	rootNode.createObject('CollisionResponse', name='Response', response='default')
	rootNode.createObject('CollisionGroup', name='Group')
	rootNode.findData('dt').value=0.05

	# floor mesh
	floorNode = rootNode.createChild('Floor')
	floorNode.createObject('MeshObjLoader', name='loader', filename='mesh/floor2b.obj')
	floorNode.createObject('Mesh', src='@loader')
	floorNode.createObject('MechanicalObject', src='@loader')
	floorNode.createObject('Triangle', name='Floor', simulated=0, moving=0)
	floorNode.createObject('Line', name='Floor', simulated=0, moving=0)
	floorNode.createObject('Point', name='Floor', simulated=0, moving=0)
	floorNode.createObject('OglModel', name='FloorV', filename='mesh/floor2b.obj', texturename='textures/floor.bmp')

	# make some dynamic meshes spawn...
	for i in xrange(-1,2):
		for j in xrange(-1, 2):
			#print 'x='+str(i)+' z='+str(j)
			color = randomColor()
			createArmadillo(rootNode,'Armadillo',i*20,50,j*20,color)


	return rootNode
