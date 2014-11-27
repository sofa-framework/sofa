import Sofa
import SofaTest

from Compliant import StructuralAPI, Tools
from Compliant.Tools import cat as concat
import random

path = Tools.path( __file__ )
AllowSleep=False
ShowObjects=False
template_suffix="d"

def createRigidBody(node,name,mesh,pos=[0,0,0],size=[1,1,1]):
	body = StructuralAPI.RigidBody( node, name )
	body.setManually( [pos[0],pos[1],pos[2],0,0,0,1], 1, [1,1,1] )
	if ShowObjects:
		body.dofs.showObject=True
		body.dofs.showObjectScale=1
	body.node.canChangeSleepingState=AllowSleep
	collisionModel = body.addCollisionMesh( mesh, size )
	body.collisionModel = collisionModel.triangles
	return body

def createRigidBoxBody(node,name,pos=[0,0,0],size=[1,1,1]):
	body = StructuralAPI.RigidBody( node, name )
	body.setManually( [pos[0],pos[1],pos[2],0,0,0,1], 1, [1,1,1] )
	if ShowObjects:
		body.dofs.showObject=True
		body.dofs.showObjectScale=1
	body.node.canChangeSleepingState=AllowSleep
	body.collisionModel = body.node.createObject('TOBBModel', name='model', template="Rigid3"+template_suffix, extents=concat(size))
	return body

def setFixed(body):
	body.node.createObject('FixedConstraint')
	body.collisionModel.moving = False
	body.collisionModel.simulated = False

def createScene(root):
	##### global parameters
	root.createObject('VisualStyle', displayFlags="hideVisual showBehavior showCollisionModels hideBoundingCollisionModels hideMapping hideOptions" )
	root.dt = 0.01
	root.gravity = [0, -9.8, 0]

	root.createObject('RequiredPlugin', pluginName = 'Compliant')
	root.createObject('CompliantAttachButtonSetting')

	##### SOLVER
	root.createObject('CompliantImplicitSolver', stabilization=1)
	root.createObject('SequentialSolver', iterations=100)
	root.createObject('LDLTResponse')
	
	##### Collisions
	root.createObject('DefaultCollisionGroupManager')
	root.createObject('NewProximityIntersection', alarmDistance=0.5, contactDistance=0.05)
	root.createObject('BruteForceDetection')
	root.createObject('DefaultContactManager', response='FrictionCompliantContact', responseParams="mu=0.01")
	root.createObject('DefaultPipeline', depth=6)

	##### Sleep
	if (AllowSleep):
		root.createObject('CompliantSleepController', printLog=1, listening=1, minTimeSinceWakeUp=1.0, immobileThreshold=0.01, rotationThreshold=0.1)
	
	##### Bodies
	ground_body = createRigidBoxBody(root, 'ground_body', [0,-3,0], [20,1,20])
	# ground_body = createRigidBody(root, 'ground_body', "mesh/floor3.obj", [0,-2,0])
	setFixed(ground_body)

	random.seed(0x12345678)
	for i in range(-3,4):
		for j in range(-3,4):
			for k in range(0,2):
				body = createRigidBoxBody(root, ('cube'+str(i)+'_'+str(j)+'_'+str(k)), [i*2.4,k*2.4,j*2.4])
				body.dofs.velocity = "" + str(random.random()*4-2) + " 0 " + str(random.random()*4-2) + " 0 0 0"
				
	script = root.createObject('PythonScriptController', filename = __file__, classname = 'Controller')				


class Controller(SofaTest.Controller):
	
	iterations = 0
	
	def onEndAnimationStep(self, dt):
		self.iterations = self.iterations + 1
		if self.iterations == 20 :
			self.sendSuccess()
