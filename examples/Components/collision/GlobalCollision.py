import Sofa
import random
from cmath import *
############################################################################################
# this is a PythonScriptController example script
############################################################################################



############################################################################################
# following defs are used later in the script
############################################################################################


# utility methods

def randomColor():
	colorRandom = random.randint(0,6)
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


def createRigidCapsule(parentNode,name,x,y,z):
	node = parentNode.createChild(name)
	radius=random.uniform(0.5,1.5)

	meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1')
	mass = node.createObject('UniformMass',name='mass',totalMass=1)

	x_rand=random.uniform(-0.5,0.5)
	y_rand=random.uniform(-0.5,0.5)
	z_rand=random.uniform(-0.5,0.5)

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MechanicalObject',template='Vec3d',name='falling_particle',position=str(x_rand)+' '+str(y_rand)+' '+str(z_rand)+' '+str(-x_rand)+' '+str(-y_rand)+' '+str(-z_rand))
	SurfNode.createObject('MeshTopology', name='meshTopology34',edges='0 1',drawEdges='1')
	SurfNode.createObject('TCapsuleModel',template='Vec3d',name='capsule_model',defaultRadius=str(radius))
	SurfNode.createObject('RigidMapping',template='Rigid,Vec3d',name='rigid_mapping',input='@../rigidDOF',output='@falling_particle')

	return 0



def createOBB(parentNode,name,x,y,z):
	node = parentNode.createChild(name)
	a=random.uniform(0.5,1.5)
	b=random.uniform(0.5,1.5)
	c=random.uniform(0.5,1.5)

	meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1')
	mass = node.createObject('UniformMass',name='mass',totalMass=1)

	node.createObject('TOBBModel',template='Rigid',name='OBB_model',extents=str(a)+' '+str(b)+' '+str(c))

	return 0


def createFlexCapsule(parentNode,name,x,y,z):
	node = parentNode.createChild(name)
	radius=random.uniform(1,3)

	x_rand=random.uniform(-0.5,0.5)
	y_rand=random.uniform(-0.5,0.5)
	z_rand=random.uniform(-0.5,0.5)

	node = node.createChild('Surf')
	node.createObject('MechanicalObject',template='Vec3d',name='falling_particle',position=str(x + x_rand)+' '+str(y + y_rand)+' '+str(z + z_rand + 20)+' '+str(x - x_rand)+' '+str(y - y_rand)+' '+str(z - z_rand))
	mass = node.createObject('UniformMass',name='mass')
	node.createObject('MeshTopology', name='meshTopology34',edges='0 1',drawEdges='1')
	node.createObject('TCapsuleModel',template='Vec3d',name='capsule_model',defaultRadius=str(radius))

	return 0

def createCapsule(parentNode,name,x,y,z):
	if random.randint(0,1) == 0:
		createRigidCapsule(parentNode,name,x,y,z)
	else:
		createFlexCapsule(parentNode,name,x,y,z)

	return 0


def createSphere(parentNode,name,x,y,z):
	node = parentNode.createChild(name)

	r=random.uniform(1,4)

	meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+
	                         str(z)+' 0 0 0 1')
	mass = node.createObject('UniformMass',name='mass',totalMass=1)

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MechanicalObject',template='Vec3d',name='falling_particle',position='0 0 0')
	SurfNode.createObject('TSphereModel',template='Vec3d',name='sphere_model',radius=str(r))
	SurfNode.createObject('RigidMapping',template='Rigid,Vec3d',name='rigid_mapping',input='@../rigidDOF',output='@falling_particle')

	return 0


############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################


class GlobalCollision(Sofa.PythonScriptController):

	# optionnally, script can create a graph...
	def createGraph(self,node):

		node.findData('dt').value=0.05
		self.timestepgap = int( 2.1 / node.findData('dt').value )

		self.rootNode = node.createChild('pythonnode')

		# a container
		floorNode = self.rootNode.createChild('Floor')
		floorNode.createObject('MeshObjLoader', name='loader', filename='mesh/SaladBowl.obj')
		floorNode.createObject('Mesh', src='@loader')
		mec=floorNode.createObject('MechanicalObject', src='@loader',name='the_bol')
		mec.applyScale(50,50,50)
		floorNode.createObject('Triangle', name='Floor', simulated=0, moving=0)
		floorNode.createObject('Line', name='Floor', simulated=0, moving=0)
		floorNode.createObject('OglModel', name='FloorV', filename='mesh/SaladBowl.obj',texturename='textures/texture.bmp')#, texturename='textures/SaladBowl$.bmp')

		self.generatePrimitives1(20)

		return 0


	def generatePrimitives1(self,nb):
		for i in range(0,nb):
			t1=random.uniform(0,6.28)
			t2=random.uniform(0,6.28)
			x=(10.0*(cos(t1) + cos(t2))/2.0).real
			y=(10.0*(sin(t1) + sin(t2))/2.0).real

			choice = random.randint(1,3)

			if choice == 1:
				createSphere(self.rootNode,str(self.nb_prim),x,y,self.current_height)
			elif choice == 2:
				createCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height)
			elif choice == 3:
				createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height)


			self.nb_prim +=1
			self.current_height += 5

		return 0


	# called on each animation step
	total_time = 0
	total_steps = 0
	nb_prim = 0
	current_height=20
	def onBeginAnimationStep(self,dt):
		return 0

	def onEndAnimationStep(self,dt):
		return 0


	def reset(self):
		self.total_time = 0;
		self.total_steps = 0;
		nb_prim = 0
		current_height = 20
		#print 'reset called (python side)'
		return 0
