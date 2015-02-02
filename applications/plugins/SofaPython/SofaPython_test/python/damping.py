import Sofa
import math
import os
import SofaTest
import sys

if len(sys.argv) != 7 :
  print "ERROR: wrong number of arguments"
 
#--------------------------------------------------------------------------------------------	
# manual


# the error is increasing with both the damping coef and the initial velocity
DAMPING_COEF = float( sys.argv[1] )
INITIAL_VELOCITY = float( sys.argv[2] )
DT = float( sys.argv[3] )

ACCEPTABLE_ERROR = float( sys.argv[4] )

# a ball
MASS = float( sys.argv[5] )  # the error is decreasing when the mass (-> inertia) increases
RADIUS = float( sys.argv[6] ) # the error is decreasing when the radius (-> inertia) increases

#--------------------------------------------------------------------------------------------	
# auto
INERTIA = 2.0*MASS*RADIUS*RADIUS/5.0
VOLUME = 4.0/3.0*math.pi*RADIUS*RADIUS*RADIUS
#--------------------------------------------------------------------------------------------	



class VerifController(SofaTest.Controller):

	def initGraph(self,node):
		self.translationdof = node.getObject( "/translation/dofs" )
		self.rotationdof = node.getObject( "/rotation/dofs" )
		self.t = 0
		self.rotationmax = 0
		self.translationmax = 0
		return 0

		
	def onEndAnimationStep(self,dt):
		self.t += dt
		
		
		rotationtheory = INITIAL_VELOCITY*math.exp(-DAMPING_COEF*self.t/INERTIA) # theoretical velocity
		translationtheory = INITIAL_VELOCITY*math.exp(-DAMPING_COEF*self.t/MASS) # theoretical velocity
		
		
		# test check
		if self.t >= 2 or rotationtheory<1e-10 or translationtheory<1e-10:
		  #print str(rotationtheory)+" "+str(translationtheory)
		  #print str(self.rotationmax)+" "+str(self.translationmax)+" "+str(ACCEPTABLE_ERROR)
		  self.should( self.rotationmax < ACCEPTABLE_ERROR and self.translationmax < ACCEPTABLE_ERROR, "angular damping error: "+str(self.rotationmax)+" translation damping error: "+str(self.translationmax) )
		  
		
		#rotation error
		rotationmeasure = self.rotationdof.velocity[0][3] # simulated velocity
		rotationerror = abs(rotationmeasure-rotationtheory)/rotationtheory
		if rotationerror>self.rotationmax :
		  self.rotationmax = rotationerror
		  
		#translation error
		translationmeasure = self.translationdof.velocity[0][0] # simulated velocity
		translationerror = abs(translationmeasure-translationtheory)/translationtheory
		if translationerror>self.translationmax :
		  self.translationmax = translationerror
		
		#print str(translationmeasure)+" "+str(translationtheory)+" "+str(translationerror)+" "+str(self.translationmax)
		
		#print str(rotationmeasure)+" "+str(rotationtheory)+" "+str(rotationerror)+" "+str(self.rotationmax)
		
		
		
            	return 0
		



		
		
		
#------------------------------------------------------------------------------------------------------------------------------------------------
def createScene(node):
  
    if DAMPING_COEF > MASS / DT :
      print "WARNING too large damping coefficient compared to time step\n"

    node.findData('dt').value=DT
    node.findData('gravity').value='0 0 0'
    
    node.createObject('VisualStyle', displayFlags='showBehaviorModels')
    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')
    
    
    # create a rigid file to give a correct inertia matrix
    path = os.path.dirname( os.path.abspath( __file__ ) )
    rigidFile = open(path+"/damping_mass.rigid", "wb")
    rigidFile.write( 'Xsp 3.0\n' )
    rigidFile.write( 'mass '+str(MASS)+'\n' )
    rigidFile.write( 'volm '+str(VOLUME)+'\n' )
    rigidFile.write( 'inrt '+str(INERTIA/MASS) + ' 0 0   0 '+str(INERTIA/MASS) + ' 0  0 0 '+str(INERTIA/MASS) + '\n' )
    rigidFile.write( 'cntr 0 0 0\n' )
    rigidFile.close()
    
    #node.createObject('EulerSolver',name='odesolver')
    #node.createObject('RequiredPlugin', pluginName = 'Compliant')
    #node.createObject('CompliantImplicitSolver',name='odesolver',stabilization='0')
    #node.createObject('LDLTSolver',name = 'numsolver')
    
    # angular damping test
    angularNode = node.createChild('rotation')
    angularNode.createObject('EulerImplicit',name='odesolver',rayleighStiffness=0,rayleighMass=0)
    angularNode.createObject('CGLinearSolver',name = 'numsolver',precision=1e-10,threshold=1e-10,iterations=1000)
    angularNode.createObject('MechanicalObject', template="Rigid", name="dofs", position="0 0 0 0 0 0 1", velocity="0 0 0 "+str(INITIAL_VELOCITY)+" 0 0")
    angularNode.createObject('UniformMass', filename=path+"/damping_mass.rigid")
    angularNode.createObject('UniformVelocityDampingForceField', dampingCoefficient=DAMPING_COEF)
    #angularNode.createObject('PartialFixedConstraint', indices='0', fixedDirections="1 1 1 0 1 1")
    
    
    # translation damping test
    translationNode = node.createChild('translation')
    translationNode.createObject('EulerImplicit',name='odesolver',rayleighStiffness=0,rayleighMass=0)
    translationNode.createObject('CGLinearSolver',name = 'numsolver',precision=1e-10,threshold=1e-10,iterations=1000)
    translationNode.createObject('MechanicalObject', template="Rigid", name="dofs", position="0 0 0 0 0 0 1", velocity=str(INITIAL_VELOCITY)+" 0 0  0 0 0")
    translationNode.createObject('UniformMass', filename=path+"/damping_mass.rigid")
    translationNode.createObject('UniformVelocityDampingForceField', dampingCoefficient=DAMPING_COEF)
    #translationNode.createObject('PartialFixedConstraint', indices='0', fixedDirections="0 1 1 1 1 1")
    
    
    return node

