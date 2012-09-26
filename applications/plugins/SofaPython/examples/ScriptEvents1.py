import Sofa

############## Python example: ScriptEvents ##############'
### This example scene demonstrates the communication between separate scripts in a Sofa scene.'
### the script1 receives mouse events, then sends a script event names "button1" or "button2"'
### script2 and script3 respectively aknowledge button1 and button2 events by printing a console message'
##########################################################'



# called once the script is loaded
def onLoaded(node):
	print '############## Python example: ScriptEvents ##############'
	print '### This example scene demonstrates the communication between separate scripts in a Sofa scene.'
	print '### the script1 receives mouse events, then sends a script event named "button1" or "button2"'
	print '### script2 and script3 respectively aknowledge button1 and button2 events by printing a console message'
	print '##########################################################'
	
	global rootNode
	rootNode = node.getRoot()
	return 0

def onMouseButtonLeft(x,y,pressed):
	print 'script1: onMouseButtonLeft x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
	data=[x,y,pressed]
	rootNode.sendScriptEvent('button1',data)
	return 0
 
def onMouseButtonRight(x,y,pressed):
	print 'script1: onMouseButtonRight x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
	data=[x,y,pressed]
	rootNode.sendScriptEvent('button2',data)
	return 0
 
 
 
