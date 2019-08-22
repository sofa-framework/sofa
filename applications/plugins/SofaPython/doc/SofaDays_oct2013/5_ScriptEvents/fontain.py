import Sofa
import ObjectCreator
import random


############################################################################################
# this class sends a script event as soon as the particle has fallen below a certain level
############################################################################################

class Particle(Sofa.PythonScriptController):
	# called once the script is loaded
	def onLoaded(self,node):
		self.myNode = node
		self.particleObject=node.getObject('MecaObject')
		return 0

	# called on each animation step
	def onBeginAnimationStep(self,dt):
		position = self.particleObject.findData('position').value
		if position[0][1]<-5.0:
			self.myNode.sendScriptEvent('below_floor',0)
		return 0


############################################################################################
# particle spawn point
############################################################################################

class Fontain(Sofa.PythonScriptController):

	# called once the script is loaded
	def onLoaded(self,node):
		self.rootNode = node
		self.particleCount = 0
		return 0

	particleCount = 0
	def spawnParticle(self,node):
		# create the particle
		node = ObjectCreator.createCube(node,'particle'+str(self.particleCount),0,0,0,random.uniform(-10,10),random.uniform(10,30),random.uniform(-10,10),'red')
		self.particleCount+=1
		# add the controller script
		node.createObject('PythonScriptController', name='script', filename='fontain.py', classname='Particle')
		return node
	 
	# optionnally, script can create a graph...
	def createGraph(self,node):
		for i in range(10):
			self.spawnParticle(self.rootNode)
		return 0

	def onScriptEvent(self,senderNode,eventName,data):
		print 'onScriptEvent eventName='+eventName+' data='+str(data)+' sender='+senderNode.name
		if eventName=='below_floor':
			self.rootNode.removeChild(senderNode)
			self.spawnParticle(self.rootNode)
		return 0
