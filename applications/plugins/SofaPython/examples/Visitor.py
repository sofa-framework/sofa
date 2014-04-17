import Sofa

############################################################################################
# this is a PythonScriptController example script
############################################################################################



############################################################################################
# following defs are used later in the script
############################################################################################


def testNodes(node):
	node.findData('name').value = 'god'

	# Node creation
	adam = node.createChild('Adam')
	eve = node.createChild('Eve')
	abel = eve.createChild('Abel')

	#you can animate simulation directly by uncommenting the following line:
	#node.animate=truee

	return 0

############################################################################################
# this is my parent visitor class
############################################################################################

class SofaVisitor(object):
	def __init__(self,name):
		print 'SofaVisitor constructor name='+name
		self.name = name

	def processNodeTopDown(self,node):
		print 'SofaVisitor "'+self.name+'" processNodeTopDown node='+node.findData('name').value
		return True

	def processNodeBottomUp(self,node):
		print 'SofaVisitor "'+self.name+'" processNodeBottomUp node='+node.findData('name').value
		
	def treeTraversal(self):
		print 'SofaVisitor "'+self.name+'" treeTraversal'
		return -1 # dag
 
  


############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################


class Visitor(Sofa.PythonScriptController):
	# called once the script is loaded
	def onLoaded(self,node):
		self.rootNode = node.getRoot()
		print 'Controller script loaded from node %s'+node.findData('name').value
		return 0

	# optionnally, script can create a graph...
	def createGraph(self,node):
		print 'createGraph called (python side)'

		#uncomment to create nodes
		testNodes(node)


		return 0



	# called once graph is created, to init some stuff...
	def initGraph(self,node):
		print 'initGraph called (python side)'

		v = SofaVisitor('PythonVisitor')
		node.executeVisitor(v)

		return 0

 
 
