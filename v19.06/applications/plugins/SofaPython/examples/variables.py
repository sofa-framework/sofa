import Sofa


class Variables(Sofa.PythonScriptController):

	def onLoaded(self,node):
	  
		print '############## Python example: variables ##############'
		print '### This example scene demonstrates how to pass string variables from a SOFA scn to the python script'
		print '### Note that the variables can be modified in the SOFA GUI'
		print '##########################################################'
		
		variables = self.findData('variables').value
		
		print 'At initialization, variables = ', variables
		
		self.timestep = 0
			
		return 0
		
		
		
		
	def onBeginAnimationStep(self,dt):
	  
		self.timestep = self.timestep+1
		
		# change the fist variable to the current time step
		variables = self.findData('variables').value
		variables[0][0] = str(self.timestep)
		self.findData('variables').value = variables
		
		
		
	def onEndAnimationStep(self,dt):
	  
		variables = self.findData('variables').value # just to be sure to consider the right variables
		print 'During simulation, variables can be modified (even manually in SOFA GUI) = ', variables
