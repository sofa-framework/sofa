import Sofa
import math
import os
import Recalage

from decimal import Decimal


runFromScript=False

if(runFromScript):
	#recuperation variable d'environement
	yM= os.environ["YoungModulus"]
	
	
	
############################################################################################
# 
############################################################################################



class RecalageController(Sofa.PythonScriptController):

	# Graph creation
	def createGraph(self,node):
		print 'createGraph called (python side)'
		multi=True
		restPosition=False
		keepMatch=False
		updateStiffness=True
		
		self.recalage=Recalage.Recalage(multi,restPosition,keepMatch,updateStiffness,runFromScript)
		
		#self.recalage.createGlobalStuff(node)
		#uncomment to create nodes
		self.recalage.createScene(node)
		#rootAndLiverNodes(self,node)

		self.rootNode = node.getRoot()
		
		return 0



	def initGraph(self,node):
		print 'initGraph called (python side)'
		self.step	= 	0
		self.total_time = 	0
		
		
		self.recalage.initializationObjects(node)
		return 0


	def reset(self):
		print 'reset called (python side)'
		return 0

	
	# called on each animation step
	def onBeginAnimationStep(self,dt):
		#print "self.step = ",self.step
		#print "self.total_time = ",self.total_time
		self.step += 1
		self.total_time += dt		
		
		#print "debut process"
		self.recalage.process(self.step,self.total_time)
		#print "fin process"
		
		return 0


