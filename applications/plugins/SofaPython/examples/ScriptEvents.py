import Sofa

############## Python example: ScriptEvents ##############'
### This example scene demonstrates the communication between separate scripts in a Sofa scene.'
### the script1 receives mouse events, then sends a script event names "button1" or "button2"'
### script2 and script3 respectively aknowledge button1 and button2 events by printing a console message'
##########################################################'

class script1(Sofa.PythonScriptController):

	# called once the script is loaded
	def onLoaded(self,node):
		print '############## Python example: ScriptEvents ##############'
		print '### This example scene demonstrates the communication between separate scripts in a Sofa scene.'
		print '### the script1 receives mouse events, then sends a script event named "button1" or "button2"'
		print '### script2 and script3 respectively aknowledge button1 and button2 events by printing a console message'
		print '##########################################################'
		
		self.rootNode = node.getRoot()
		return 0

	def onMouseButtonLeft(self,x,y,pressed):
		print 'script1: onMouseButtonLeft x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
		data=[x,y,pressed]
		self.rootNode.sendScriptEvent('button1',data)
		return 0
	 
	def onMouseButtonRight(self,x,y,pressed):
		print 'script1: onMouseButtonRight x='+str(x)+' y='+str(y)+' pressed='+str(pressed)
		data=[x,y,pressed]
		self.rootNode.sendScriptEvent('button2',data)
		return 0
	 
	 
class script2(Sofa.PythonScriptController):
 
	def onScriptEvent(self,senderNode,eventName,data):
		if eventName=='button1':
			print 'script2 received a "button1" script event from '+senderNode.name+'; data='+str(data)
		return 0


class script3(Sofa.PythonScriptController):

	def onScriptEvent(self,senderNode,eventName,data):
		if eventName=='button2':
			print 'script3 received a "button2" script event from '+senderNode.name+'; data='+str(data)
		return 0
