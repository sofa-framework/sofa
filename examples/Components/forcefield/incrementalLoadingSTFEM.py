import Sofa
import sys

############################################################################################
# following defs are optionnal entry points, called by the PythonScriptController component;
############################################################################################
class IncrementalLoading(Sofa.PythonScriptController):
	# called once the script is loaded
	def initGraph(self,node):
		self.root = node
		self.dt = self.root.findData('dt').value
		self.STFEMFF = self.root.getChild('MeshTopology').getChild('StandardTetrahedralFEMForceField')
		self.CFF = self.STFEMFF.getObject('CFF')
		self.forceFinal = self.CFF.findData('force').value
		self.forceFinal[0][0] = 0.0001
		self.forceFinal[0][1] = 0
		self.forceFinal[0][2] = 0
		print self.forceFinal
		return 0

	# # optionnally, script can create a graph...
	# def createGraph(self,node):
	# 	print 'createGraph called (python side)'
	# 	return 0

	# # called once graph is created, to init some stuff...
	# def initGraph(self,node):
	# 	#print(str(dir(self.__class__)))
	# 	return 0

	# called on each animation step
	def onBeginAnimationStep(self,dt):
		if self.root.findData('time').value < 1.0:
			incrementalFactor = self.root.findData('time').value
			force = self.CFF.findData('force').value
			force[0][0] = self.forceFinal[0][0] * incrementalFactor
			force[0][1] = self.forceFinal[0][1] * incrementalFactor
			force[0][2] = self.forceFinal[0][2] * incrementalFactor
			self.CFF.findData('force').value = force
			print force
		else:
			self.CFF.findData('force').value = self.forceFinal
		return 0