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

falling_speed = 0
capsule_height = 5
capsule_chain_height = 5

def createRigidCapsule(parentNode,name,x,y,z,*args):
	node = parentNode.createChild(name)
	radius=0
	if len(args)==0:
		radius = random.uniform(1,3)
	else:
		radius = args[0]

	meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1',velocity='0 0 '+str(falling_speed)+' 0 0 0 1')
	mass = node.createObject('UniformMass',name='mass',totalMass=1)

	x_rand=random.uniform(-0.5,0.5)
	y_rand=random.uniform(-0.5,0.5)
	z_rand=random.uniform(-0.5,0.5)

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MechanicalObject',template='Vec3d',name='falling_particle',position=str(x_rand)+' '+str(y_rand)+' '+str(capsule_height/2)+' '+str(-x_rand)+' '+str(-y_rand)+' '+str(- capsule_height/2))
	SurfNode.createObject('MeshTopology', name='meshTopology34',edges='0 1',drawEdges='1')
	SurfNode.createObject('TCapsuleModel',template='Vec3d',name='capsule_model',defaultRadius=str(radius))
	SurfNode.createObject('RigidMapping',template='Rigid,Vec3d',name='rigid_mapping',input='@../rigidDOF',output='@falling_particle')

	return node


def createFlexCapsule(parentNode,name,x,y,z,*args):
	radius=0
	if len(args)==0:
		radius = random.uniform(1,3)
	else:
		radius = args[0]

	node = parentNode.createChild(name)

	x_rand=random.uniform(-0.5,0.5)
	y_rand=random.uniform(-0.5,0.5)
	z_rand=random.uniform(-0.5,0.5)

	node = node.createChild('Surf')
	node.createObject('MechanicalObject',template='Vec3d',name='falling_particle',position=str(x + x_rand)+' '+str(y + y_rand)+' '+str(z + z_rand + capsule_height)+' '+str(x - x_rand)+' '+str(y - y_rand)+' '+str(z - z_rand),velocity='0 0 '+str(falling_speed))
	mass = node.createObject('UniformMass',name='mass')
	node.createObject('MeshTopology', name='meshTopology34',edges='0 1',drawEdges='1')
	node.createObject('TCapsuleModel',template='Vec3d',name='capsule_model',defaultRadius=str(radius))

	return node


def createCapsuleChain(parentNode,name,length,x,y,z):
	node = parentNode.createChild(name)

	#radius=random.uniform(1,3)
	radius=0.5
	height=5

	x_rand=random.uniform(-0.5,0.5)
	y_rand=random.uniform(-0.5,0.5)
	z_rand=random.uniform(-0.5,0.5)

	node = node.createChild('Surf')

	ray = 3.0
	t = 0.0
	delta_t = 0.7
	topo_edges=''
	particles=''
	velocities = ''
	springs=''
	for i in range(0,length):
		particles += str(x + (ray * cos(t)).real)+' '+str(y + (ray * sin(t)).real)+' '+str(z + i*capsule_chain_height)+' '
		t += delta_t

		if i < length -1:
			topo_edges += str(i)+' '+str(i + 1)+' '
			springs += str(i)+' '+str(i + 1)+' 10 1 '+str(capsule_chain_height)+' '

		velocities+='0 0 '+str(falling_speed)+' '



	topo_edges += str(length - 2)+' '+str(length -1)
	springs += str(length - 2)+' '+str(length -1)+' 10 1 '+str(capsule_chain_height)

	node.createObject('MechanicalObject',template='Vec3d',name='falling_particles',position=particles,velocity=velocities)
	node.createObject('StiffSpringForceField',template='Vec3d',name='springforcefield',stiffness='100',damping='1',spring=springs)
	mass = node.createObject('UniformMass',name='mass')
	node.createObject('MeshTopology', name='meshTopology34',edges=topo_edges,drawEdges='1')
	node.createObject('TCapsuleModel',template='Vec3d',name='capsule_model',defaultRadius=str(radius))

	return node


def createOBB(parentNode,name,x,y,z,*args):
	a=0
	b=0
	c=0
	if len(args)==0:
		a=random.uniform(0.5,1.5)
		b=random.uniform(0.5,1.5)
		c=random.uniform(0.5,1.5)
	else:
		a=args[0]
		b=args[1]
		c=args[2]

	node = parentNode.createChild(name)

	meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1',velocity='0 0 '+str(falling_speed)+' 0 0 0 1')
	mass = node.createObject('UniformMass',name='mass',totalMass=1)

	node.createObject('TOBBModel',template='Rigid',name='OBB_model',extents=str(a)+' '+str(b)+' '+str(c))

	return node

def createCapsule(parentNode,name,x,y,z):
	if random.randint(0,1) == 0:
		return createRigidCapsule(parentNode,name,x,y,z)
	else:
		return createFlexCapsule(parentNode,name,x,y,z)



def createCapsule(parentNode,name,x,y,z):
	if random.randint(0,1) == 0:
		return createRigidCapsule(parentNode,name,x,y,z)
	else:
		return createFlexCapsule(parentNode,name,x,y,z)



def createSphere(parentNode,name,x,y,z,*args):
	node = parentNode.createChild(name)

	r = 0
	if len(args) == 0:
		r=random.uniform(1,4)
	else:
		r = args[0]

	#meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+
	#                         str(z)+' 0 0 0 1')

	#SurfNode = node.createChild('Surf')
	node.createObject('MechanicalObject',template='Vec3d',name='falling_particle',position=str(x)+' '+str(y)+' '+str(z),velocity='0 0 '+str(falling_speed))
	node.createObject('SphereCollisionModel',template='Vec3d',name='sphere_model',radius=str(r))
	node.createObject('UniformMass',name='mass',totalMass=1)
	#SurfNode.createObject('RigidMapping',template='Rigid,Vec3d',name='rigid_mapping',input='@../rigidDOF',output='@falling_particle')

	return node


def createRigidSphere(parentNode,name,x,y,z,*args):
	node = parentNode.createChild(name)

	r = 0
	if len(args) == 0:
		r=random.uniform(1,4)
	else:
		r = args[0]

	#meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position=str(x)+' '+str(y)+' '+
	#                         str(z)+' 0 0 0 1')

	#SurfNode = node.createChild('Surf')
	node.createObject('MechanicalObject',template='Rigid',name='falling_particle',position=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1',velocity='0 0 '+str(falling_speed)+' 0 0 0 1')
	node.createObject('SphereCollisionModel',template='Rigid',name='sphere_model',radius=str(r))
	node.createObject('UniformMass',name='mass',totalMass=1)
	#SurfNode.createObject('RigidMapping',template='Rigid,Vec3d',name='rigid_mapping',input='@../rigidDOF',output='@falling_particle')

	return node
