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

 # Node manipulation
 eve.removeChild(abel)
 adam.addChild(abel)
 cleese.moveChild(gilliam)
 
 #you can animate simulation directly by uncommenting the following line:
 #node.animate=truee
    
 return 0

############################################################################################
# this is my visitor class
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
 
  


############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################


class Tuto6(Sofa.PythonScriptController):

	def createGraph(self,node):
	 print 'createGraph called (python side)'
	 
	 #uncomment to create nodes
	 testNodes(node)

	    
	 return 0



	def initGraph(self,node):
	 print 'initGraph called (python side)'
	 
	 v = SofaVisitor('PythonVisitor')
	 node.executeVisitor(v)
	 
	 return 0

