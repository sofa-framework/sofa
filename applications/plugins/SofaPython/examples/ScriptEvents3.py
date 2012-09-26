import Sofa

############## Python example: ScriptEvents ##############'
### This example scene demonstrates the communication between separate scripts in a Sofa scene.'
### the script1 receives mouse events, then sends a script event names "button1" or "button2"'
### script2 and script3 respectively aknowledge button1 and button2 events by printing a console message'
##########################################################'

def onLoaded(node):
	return 0

def onScriptEvent(senderName,eventName,data):
	if eventName=='button2':
		print 'script3 received a "button2" script event from '+senderName+'; data='+str(data)
	return 0
