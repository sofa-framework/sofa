import Sofa
import Visitors

# Directed Acyclic Graph test script
 
# init a DAG graph
def createDAGGraph(node):
	A0 = node.createChild('A0_Python')
	A1 = node.createChild('A1_Python')
	A2 = node.createChild('A2_Python')
	
	B0 = A1.createChild('B0_Python')
	A0.addChild(B0)

	C0 = B0.createChild('C0_Python')
	C1 = B0.createChild('C1_Python')
	A2.addChild(C1)
#	A1.addChild(C1)
	node.addChild(C1)
	
	D0 = C1.createChild('D0_Python')
	D1 = C1.createChild('D1_Python')


################################################
# visitors classes
################################################

# just go through the graph and log the nodes names
class GraphLogVisitor(object):
	def __init__(self):
	 	print 'GraphLogVisitor constructor'
  
	def processNodeTopDown(self,node):
		print 'GraphLogVisitor.processNodeTopDown node = '+node.findData('name').value
		parents = node.getParents()
		for parent in parents:
			print '    parent = '+parent.findData('name').value
		children = node.getChildren()
		for child in children:
			print '        child = '+child.findData('name').value
		return True
 
	def processNodeBottomUp(self,node):
		print 'GraphLogVisitor.processNodeBottomUp node='+node.findData('name').value
	   


################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
################################################



# called once the script is loaded
def onLoaded(node):
	global rootNode
	rootNode = node.getRoot()
	print 'Controller script loaded from node %s'+node.findData('name').value
	return 0

# optionnally, script can create a graph...
def createGraph(node):
	print 'createGraph called (python side)'
	
	#createDAGGraph(node)
	   
	return 0



# called once graph is created, to init some stuff...
def initGraph(node):
	print 'initGraph called (python side)'
	
#	v = GraphLogVisitor()
#	node.executeVisitor(v)
	
	return 0
	


# called on each animation step
def onBeginAnimationStep(dt):
	print 'onBeginAnimationStep '+str(dt)
	return 0
 
def onEndAnimationStep(dt):
	print 'onEndAnimationStep '+str(dt)
	return 0
 
# called when necessary by Sofa framework... 
def storeResetState():
	print 'storeResetState called (python side)'
	v = Visitors.GraphLogVisitor()
	rootNode.executeVisitor(v)
	return 0

def reset():
	print 'reset called (python side)'
	return 0

def cleanup():
	print 'cleanup called (python side)'
	return 0
 
 
# called when a GUIEvent is received
def onGUIEvent(controlID,valueName,value):
	print 'GUIEvent received: controldID='+controlID+' valueName='+valueName+' value='+value
	return 0 
 
 
 
# key and mouse events; use this to add some user interaction to your scripts 
def onKeyPressed(k):
	print 'onKeyPressed '+k
	return 0 
 
def onKeyReleased(k):
	print 'onKeyReleased '+k
	return 0 

def onMouseButtonLeft(x,y,pressed):
	print 'onMouseButtonLeft x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
	return 0
 
def onMouseButtonRight(x,y,pressed):
	print 'onMouseButtonRight x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
	
	if pressed:
		rootNode.simulationStep(0.42)
	return 0
 
def onMouseButtonMiddle(x,y,pressed):
	print 'onMouseButtonMiddle x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
	return 0
 
def onMouseWheel(x,y,delta):
	print 'onMouseButtonWheel x='+str(x)+' y='+str(y)+' delta='+str(delta)
	return 0
