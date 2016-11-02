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
		self.timestepgap = int( 2.1 / node.findData('dt').value )

		self.rootNode = node.createChild('pythonnode')

		# a container
		floorNode = self.rootNode.createChild('Floor')
		floorNode.createObject('MeshObjLoader', name='loader', filename='mesh/SaladBowl.obj')
		floorNode.createObject('Mesh', src='@loader')
		mec=floorNode.createObject('MechanicalObject', src='@loader',name='the_bol')
		floorNode.createObject("UniformMass", name="floorMass")
		mec.applyScale(50,50,50)
		floorNode.createObject('Triangle', name='Floor', simulated=0, moving=0)
		#floorNode.createObject('Line', name='Floor', simulated=0, moving=0)
		floorNode.createObject('OglModel', name='FloorV', filename='mesh/SaladBowl.obj',texturename='textures/texture.bmp')#, texturename='textures/SaladBowl$.bmp')
		floorNode.createObject("FixedConstraint", name="fixedConstraint", fixAll=1)
		#self.generatePrimitives1(40)

		return 0


	def genRandPrim(self):
		t1=random.uniform(0,6.28)
		t2=random.uniform(0,6.28)
		x=(10.0*(cos(t1) + cos(t2))/2.0).real
		y=(10.0*(sin(t1) + sin(t2))/2.0).real

		choice = random.randint(1,3)

		n=None

		if choice == 1:
			n = createSphere(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)
		elif choice == 2:
			n = createFlexCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,1)
		elif choice == 3:
			n = createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)

		#createFlexCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,2)
			#createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)
			#createCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height)

		n.init()

		self.nb_prim +=1
		#self.current_height += 5


	def generatePrimitives1(self,nb):
		for i in range(0,40):
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

			n = createFlexCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height,2)
			#createOBB(self.rootNode,str(self.nb_prim),x,y,self.current_height,1,1,1)
			#createCapsule(self.rootNode,str(self.nb_prim),x,y,self.current_height)

			n.init()

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
		if self.total_steps%self.timestepgap==0 and self.total_time < self.max_nb_prim :
			n = self.genRandPrim()

		return 0


	def reset(self):
		self.total_time = 0;
		self.total_steps = 0;
		self.timestepgap = 30
		nb_prim = 0
		current_height = 20
		#print 'reset called (python side)'
		return 0
