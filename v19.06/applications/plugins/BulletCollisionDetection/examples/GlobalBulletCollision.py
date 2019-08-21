import Sofa
import random
from PrimitiveCreation import *
from cmath import *

############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################


class GlobalCollision(Sofa.PythonScriptController):

	# optionnally, script can create a graph...
	def createGraph(self,node):

		node.findData('dt').value=0.05
		#self.timestepgap = int( 2.1 / node.findData('dt').value )

		self.rootNode = node.createChild('pythonnode')

		# a container
		floorNode = self.rootNode.createChild('Floor')
		rigid_meca = floorNode.createObject('MechanicalObject',name='father',template='Rigid',position='0 0 0 0 0 0 1')
		floorNode.createObject('FixedConstraint',template='Rigid')
		#floorNode.createObject('UniformMass',template='Rigid',totalMass=1)
		mapped = floorNode.createChild('mapped')
		mapped.createObject('MeshObjLoader', name='loader', filename='mesh/SaladBowl.obj')
		mapped.createObject('Mesh', src='@loader')
		mec=mapped.createObject('MechanicalObject', src='@loader',name='the_bol')
		mec.applyScale(50,50,50)
		mapped.createObject('BulletTriangleModel', name='Floor', simulated=0, moving=0, margin="1")
		#floorNode.createObject('Line', name='Floor', simulated=0, moving=0)
		mapped.createObject('OglModel', name='FloorV', filename='mesh/SaladBowl.obj',texturename='textures/texture.bmp')#, texturename='textures/SaladBowl$.bmp')
		mapped.createObject('RigidMapping',template='Rigid,Vec3d',name='rigid_mapping',input='@../father',output='@the_bol')
		#self.generatePrimitives1(40)

		# node = self.rootNode.createChild('Floor')

		# meca = node.createObject('MechanicalObject',name='rigidDOF',template='Rigid',position='0 0 0 0 0 0 1')
		# mass = node.createObject('UniformMass',name='mass',totalMass=1,template='Rigid')
		# node.createObject('FixedConstraint',template='Rigid')

		# node.createObject('BulletOBBModel',template='Rigid',name='BASE',extents='15 15 0.2',margin="0.5")

		#self.genRandPrim()
		createBulletOBB(self.rootNode,str(self.nb_prim),0,0,self.current_height,1,1,1)
		#createBulletCapsule(self.rootNode,str(self.nb_prim),5,5,self.current_height,3)

		return 0


	def genRandPrim(self):
		t1=random.uniform(0,6.28)
		t2=random.uniform(0,6.28)
		x=(10.0*(cos(t1) + cos(t2))/2.0).real
		y=(10.0*(sin(t1) + sin(t2))/2.0).real

		choice = random.randint(1,3)


		# createSphere(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)

		# if choice == 1:
		# 	createBulletSphere(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)
		# elif choice == 2:
		# 	createBulletOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)
		# else:
		# 	createBulletCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)

		createBulletOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)

		#createBulletOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)

		# elif choice == 2:
		# 	createFlexCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)

		#createFlexCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,2)
			#createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)
			#createCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height)

		self.nb_prim +=1
		#self.current_height += 5


	def generatePrimitives1(self,nb):
		for i in range(0,1):
			t1=random.uniform(0,6.28)
			t2=random.uniform(0,6.28)
			x=(10.0*(cos(t1) + cos(t2))/2.0).real
			y=(10.0*(sin(t1) + sin(t2))/2.0).real

			choice = random.randint(1,3)

			# if choice == 1:
			# 	createSphere(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)
			# elif choice == 2:
			# 	createFlexCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,2)
			# elif choice == 3:
			# 	createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)

			#createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)
			#createCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height)

			self.nb_prim +=1
			self.current_height += 5

		return 0


	# called on each animation step
	total_time = 0
	total_steps = 0
	timestepgap = 30
	nb_prim = 0
	current_height=20
	max_nb_prim = 40

	def onBeginAnimationStep(self,dt):
		self.total_time += dt
		self.total_steps += 1
		return 0

	def onEndAnimationStep(self,dt):
		if self.total_steps%self.timestepgap==0 and self.nb_prim < self.max_nb_prim :
			self.genRandPrim()

		return 0


	def reset(self):
		self.total_time = 0;
		self.total_steps = 0;
		self.timestepgap = 60
		nb_prim = 0
		current_height = 20
		#print 'reset called (python side)'
		return 0
