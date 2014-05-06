import Sofa
import math
import os
import SofaTest

#--------------------------------------------------------------------------------------------	

DAMPING_COEF = 0.5
INITIAL_VELOCITY = 10
INERTIA = 5
ACCEPTABLE_ERROR = 2e-3

#--------------------------------------------------------------------------------------------	



class VerifController(SofaTest.Controller):

	def initGraph(self,node):
		self.dof = node.getObject( "/dofs" )
		self.t = 0
		self.max = 0
		return 0

		
	def onEndAnimationStep(self,dt):
		self.t += dt
		measure = self.dof.velocity[0][3] # simulated velocity
		theory = INITIAL_VELOCITY*math.exp(-DAMPING_COEF*self.t/INERTIA) # theoretical velocity
		error = abs(measure-theory)
		if error>self.max :
		  self.max = error
		
		#print str(measure)+" "+str(theory)+" "+str(error)+" "+str(self.max)
		
		if self.t >= 10 :
		  self.should( self.max < ACCEPTABLE_ERROR )
		  
		
            	return 0
		



		
		
		
#------------------------------------------------------------------------------------------------------------------------------------------------
def createScene(node):

    node.findData('dt').value=0.01
    node.findData('gravity').value='0 0 0'

    node.createObject('EulerImplicit',name='odesolver',rayleighStiffness=0,rayleighMass=0)
    node.createObject('CGLinearSolver',name = 'numsolver',precision=1e-10,threshold=1e-10,iterations=100)
    
    
    # create a rigid file to give a correct inertia matrix
    path = os.path.dirname( os.path.abspath( __file__ ) )
    rigidFile = open(path+"/damping_mass.rigid", "wb")
    rigidFile.write( 'mass 1\n' )
    rigidFile.write( 'volm 1\n' )
    rigidFile.write( 'inrt '+str(INERTIA) + ' 0 0   0 1 0  0 0 1\n' )
    rigidFile.write( 'cntr 0 0 0\n' )
    rigidFile.close()
    
    dof = node.createObject('MechanicalObject', template="Rigid", name="dofs", position="0 0 0 0 0 0 1", velocity="0 0 0 "+str(INITIAL_VELOCITY)+" 0 0")
    #node.createObject('RigidMass', mass="1", inertia=str(INERTIA)+" "+str(INERTIA)+" "+str(INERTIA))
    node.createObject('UniformMass', filename=path+"/damping_mass.rigid")
    node.createObject('UniformVelocityDampingForceField', dampingCoefficient=DAMPING_COEF)
    node.createObject('PartialFixedConstraint', indices='0', fixedDirections="1 1 1 0 1 1")
    
    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')
    
    return node

