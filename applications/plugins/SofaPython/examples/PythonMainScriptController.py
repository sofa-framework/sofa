import Sofa
import sys

############################################################################################
# this is a PythonMainScriptController example script
############################################################################################


#first version, the controller is automatically added at the root level
def createSceneAndController(root):
    
        print 'createSceneAndController (python side)'
        sys.stdout.flush()

        return 0



# second version, adding manually the controller (in any node)
#def createScene(root):
    
	#root.createObject('PythonMainScriptController',filename=__file__)
	
	#print 'createScene (python side)'
        #sys.stdout.flush()

	#return 0



############################################################################################
# following defs are optionnal entry points, called by the PythonMainScriptController component;
############################################################################################

# called once the script is loaded
def onLoaded(node):
        print 'onLoaded called (python side) from node %s'%node.findData('name').value
        sys.stdout.flush()
        return 0

# optionnally, script can create a graph...
def createGraph(node):
        print 'createGraph called (python side)'
        sys.stdout.flush()
        return 0



# called once graph is created, to init some stuff...
def initGraph(node):
        print 'initGraph called (python side)'
        sys.stdout.flush()
        return 0

def bwdInitGraph(node):
        print 'bwdInitGraph called (python side)'
        sys.stdout.flush()
        return 0



# called on each animation step
def onBeginAnimationStep(dt):
        #print 'onBeginAnimationStep called (python side)'
        #sys.stdout.flush()
        return 0

def onEndAnimationStep(dt):
        #print 'onEndAnimationStep called (python side)'
        #sys.stdout.flush()
        return 0

# called when necessary by Sofa framework... 
def storeResetState():
        print 'storeResetState called (python side)'
        sys.stdout.flush()
        return 0

def reset():
        print 'reset called (python side)'
        sys.stdout.flush()
        return 0

def cleanup():
        print 'cleanup called (python side)'
        sys.stdout.flush()
        return 0


# called when a GUIEvent is received
def onGUIEvent(controlID,valueName,value):
        print 'GUIEvent received: controldID='+controlID+' valueName='+valueName+' value='+value
        sys.stdout.flush()
        return 0 



# key and mouse events; use this to add some user interaction to your scripts 
def onKeyPressed(k):
        print 'onKeyPressed '+k
        sys.stdout.flush()
        return 0 

def onKeyReleased(k):
        print 'onKeyReleased '+k
        sys.stdout.flush()
        return 0 

def onMouseButtonLeft(x,y,pressed):
        print 'onMouseButtonLeft x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
        sys.stdout.flush()
        return 0

def onMouseButtonRight(x,y,pressed):
        print 'onMouseButtonRight x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
        sys.stdout.flush()
        return 0

def onMouseButtonMiddle(x,y,pressed):
        print 'onMouseButtonMiddle x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
        sys.stdout.flush()
        return 0

def onMouseWheel(x,y,delta):
        print 'onMouseButtonWheel x='+str(x)+' y='+str(y)+' delta='+str(delta)
        sys.stdout.flush()
        return 0


def draw():
        #print "draw called (python side)"
        #sys.stdout.flush()
        return 0

