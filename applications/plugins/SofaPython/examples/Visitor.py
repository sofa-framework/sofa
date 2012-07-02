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
 
  


############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################



# called once the script is loaded
def onLoaded(node):
 global rootNode
 rootNode = node.getRoot()
 print 'Controller script loaded from node %s'+node.findData('name').value
 return 0

# optionnally, script can create a graph...
def createGraph(node):
 print 'createGraph called (python side)'
 
 #uncomment to create nodes
 testNodes(node)

    
 return 0



# called once graph is created, to init some stuff...
def initGraph(node):
 print 'initGraph called (python side)'
 
 v = SofaVisitor('testVisitor')
 node.executeVisitor(v)
 
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
 
 
 
