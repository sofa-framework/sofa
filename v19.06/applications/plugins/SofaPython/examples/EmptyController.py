import Sofa
import sys

############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################
class ExampleController(Sofa.PythonScriptController):
	# called once the script is loaded
	def onLoaded(self,node):
		self.counter = 0
		print 'Controller script loaded from node %s'%node.findData('name').value
		return 0

	# optionnally, script can create a graph...
	def createGraph(self,node):
		print 'createGraph called (python side)'
		return 0

	# called once graph is created, to init some stuff...
	def initGraph(self,node):
		#print(str(dir(self.__class__)))
		return 0

	def bwdInitGraph(self,node):
		#print 'bwdInitGraph called (python side)'
		sys.stdout.flush()
		return 0

	# called on each animation step
	#total_time = 0
        def onBeginAnimationStep(self,dt):
	#	print 'onBeginAnimatinStep (python) dt=%f total time=%f'%(dt,self.total_time)
		return 0

	def onEndAnimationStep(self,dt):
	        return 0

	# called when necessary by Sofa framework... 
	def storeResetState(self):
		return 0

	def reset(self):
		return 0

	def cleanup(self):
		return 0


	# called when a GUIEvent is received
	def onGUIEvent(self,controlID,valueName,value):
		return 0 

	# key and mouse events; use this to add some user interaction to your scripts 
	def onKeyPressed(self,k):
	        print("TOTO")
		return 0 

	def onKeyReleased(self,k):
		return 0 

	def onMouseMove(self, x, y):
	        print("MOUSE MOVE...")
	        return 0                

	def onMouseButtonLeft(self,x,y,pressed):
		return 0

	def onMouseButtonRight(self,x,y,pressed):
		return 0

	def onMouseButtonMiddle(self,x,y,pressed):
		return 0
        
	def onMouseWheel(self,x,y,delta):
		return 0

	# called at each draw (possibility to use PyOpenGL)
	def draw(self):
		return 0
 
