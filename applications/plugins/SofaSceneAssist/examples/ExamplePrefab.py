import Sofa
import sys

class ExamplePrefab(Sofa.PythonScriptController):
	# called once the script is loaded
	def onLoaded(self,node):
		self.counter = 0
		print 'Controller script loaded from node %s'%node.findData('name').value
		return 0

	# optionnally, script can create a graph...
	def onPrefabChanged(self,node):
		print 'createGraph called (python side)'
		
		for i in range(0, 10)
			node.createObject("File "+10)
			
		return 0

	# called once graph is created, to init some stuff...
	def initGraph(self,node):
		print 'initGraph called (python side)'
		return 0

	def bwdInitGraph(self,node):
		print 'bwdInitGraph called (python side)'
		sys.stdout.flush()
		return 0

	def update(self):
		return 
