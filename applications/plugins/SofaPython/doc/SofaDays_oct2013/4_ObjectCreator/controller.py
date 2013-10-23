import Sofa
import ObjectCreator

# this "thing" creates and throws balls

class Tuto4(Sofa.PythonScriptController):

	# called once graph is created, to init some stuff...
	def createGraph(self,node):
		global rootNode
		rootNode = node
		return 0 
	 
	# key and mouse events; use this to add some user interaction to your scripts 
	def onKeyPressed(self,k):
		if k=='A':
			print 'Armadillo, go!'
			ObjectCreator.createArmadillo(rootNode,'RedArmadillo',0,50,0,'red')
		if k=='D':
			print 'Dragon launched!'
			obj = ObjectCreator.createDragon(rootNode,'GreenDragon',0,50,0,'green')
			#obj.findData('restVelocity').value=[0.0, 20.0, 0.0 ]
		return 0 
	 
 