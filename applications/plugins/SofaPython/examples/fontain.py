import Sofa
import random

############################################################################################
# in this sample, a controller spawns particles and is responsible to delete them when necessary, then re-spawn others
# Each particle has a script itself that check the particle's altitude an sends a message to the fontain script
# when it reachs the minimum altitude, so that it can be removed from the scene.
############################################################################################

class Fontain(Sofa.PythonScriptController):


	def createCube(self,parentNode,name,x,y,z,vx,vy,vz,color):
		node = parentNode.createChild(name)

		node.createObject(Sofa.BaseObjectDescription('cg_odesolver','EulerImplicit'))

		desc = Sofa.BaseObjectDescription('linear solver','CGLinearSolver')
		desc.setAttribute('iterations','25')
		desc.setAttribute('tolerance','1.0e-9')
		desc.setAttribute('threshold','1.0e-9')
		node.createObject(desc)

		desc = Sofa.BaseObjectDescription('MecaObject','MechanicalObject')
		desc.setAttribute('template','Rigid')
		object = node.createObject(desc)

		mass = node.createObject(Sofa.BaseObjectDescription('mass','UniformMass'))
		mass.findData('totalmass').value=100

		# VisualNode
		VisuNode = node.createChild('Visu')

		desc = Sofa.BaseObjectDescription('Visual','OglModel')
		desc.setAttribute('fileMesh','mesh/PokeCube.obj')
		desc.setAttribute('color',color)
		VisuNode.createObject(desc)

		desc = Sofa.BaseObjectDescription('mapping','RigidMapping')
		desc.setAttribute('object1','@..')
		desc.setAttribute('object2','@Visual')
		VisuNode.createObject(desc)

		# apply wanted initial translation
		#object.applyTranslation(x,y,z)
		object.findData('position').value=str(x)+' '+str(y)+' '+str(z)+' 0 0 0 1'
		object.findData('velocity').value=str(vx)+' '+str(vy)+' '+str(vz)+' 0 0 0'
		
		return node

	
	# called once the script is loaded
	def onLoaded(self,node):
		print 'Fontain.onLoaded called from node '+node.name
		self.rootNode = node
	
	particleCount = 0
	def spawnParticle(self):
		# create the particle, with a random color
		color='red'
		colorRandom = random.randint(1,6)
		if colorRandom==1:
			color = 'red'
		if colorRandom==2:
			color = 'green'
		if colorRandom==3:
			color = 'blue'
		if colorRandom==4:
			color = 'yellow'
		if colorRandom==5:
			color = 'cyan'
		if colorRandom==6:
			color = 'magenta'
		node = self.createCube(self.rootNode,'particle'+str(self.particleCount),0,0,0,random.uniform(-10,10),random.uniform(10,30),random.uniform(-10,10),color)
		self.particleCount+=1
		# add the controller script
		desc=Sofa.BaseObjectDescription('script','PythonScriptController')
		desc.setAttribute('filename','fontain.py')
		desc.setAttribute('classname','Particle')
		node.createObject(desc)
	 
	# optionnally, script can create a graph...
	def createGraph(self,node):
		print 'Fontain.createGraph called from node '+node.name	
		for i in range(1,100):
			self.spawnParticle()
	
	def onScriptEvent(self,senderNode,eventName,data):
		print 'onScriptEvent eventName='+eventName+' data='+str(data)+' sender='+senderNode.name
		if eventName=='below_floor':
			self.rootNode.removeChild(senderNode)
			self.spawnParticle()





############################################################################################
# this class sends a script event as soon as the particle has fallen below a certain level
############################################################################################
class Particle(Sofa.PythonScriptController):
	# called once the script is loaded
	def onLoaded(self,node):
		self.myNode = node
		self.particleObject=node.getObject('MecaObject')
	
	# called on each animation step
	def onBeginAnimationStep(self,dt):
		position = self.particleObject.findData('position').value
		if position[1]<-5.0:
			self.myNode.sendScriptEvent('below_floor',0)
		return 0
