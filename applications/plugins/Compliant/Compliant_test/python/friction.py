import Sofa
import SofaTest
from SofaTest.Macro import *

import math

from Compliant import Rigid, Vec, Tools, Control, StructuralAPI
from SofaPython import Quaternion
import numpy
import random
import sys


class Shared:
    pass

global shared
shared = Shared()

dir = Tools.path( __file__ )

def createScene(node):
    # controller
    node.createObject('PythonScriptController', filename = __file__, classname = 'Controller' )


    # friction coefficient
    shared.mu = float( random.randint(0,10) ) / 10.0 # a random mu in [0,1] with 0.1 step

    scene = Tools.scene( node )
    node.dt = 0.005

    style = node.getObject('style')
    style.findData('displayFlags').showMappings = True

    manager = node.getObject('manager')
    manager.response = 'FrictionCompliantContact'
    manager.responseParams = 'mu=' + str(shared.mu) +"&horizontalConeProjection=1"

    ode = node.getObject('ode')
    ode.stabilization = "pre-stabilization"
    
    num = node.createObject('SequentialSolver',
                            name = 'num',
                            iterations = 1000,
                            precision = 1e-20)
    
    proximity = node.getObject('proximity')

    proximity.alarmDistance = 0.5
    proximity.contactDistance = 0.1


    # plane
    plane = StructuralAPI.RigidBody( node, 'plane' )
    plane.setManually( [0,0,0,0,0,0,1], 1, [1,1,1] )
    plane.node.createObject('FixedConstraint')
    cm = plane.addCollisionMesh( "mesh/cube.obj", [10,1,10] )
    cm.addVisualModel()
    
    # box
    box = StructuralAPI.RigidBody( node, 'box' )
    box.setFromMesh( 'mesh/cube.obj', 50, [0,2.5,0,0,0,0,1] )
    cm = box.addCollisionMesh( "mesh/cube.obj" )
    cm.addVisualModel()

    # keep an eye on dofs
    shared.plane = plane.dofs
    shared.box = box.dofs



# scene controller
class Controller(SofaTest.Controller):

    muAngle = 0 # the plane angle corresponding to given mu
    currentAngle = 0 # the current plane angle
    muToTest = 0.1 # stop at this mu to see if the box is sticking or sliding
    
    counter = 0 # to wait at a given mu

    def reset(self):
        self.muAngle = math.atan(shared.mu)
        return 0

    
    def onBeginAnimationStep(self, dt):

        # current mu from current plane angle
        currentMu = math.tan( self.currentAngle )

        # is it a mu we want to test?
        if numpy.allclose( self.muToTest, currentMu, 1e-3, 1e-3 ) :
            if self.counter < 100 : # does not rotate the plane for 100 time steps
                self.counter += 1
                return 0
            else:
                # at the end of 100 time steps, check if the box was sticking or sliding
                self.counter = 0
                self.muToTest += 0.1
                
                # look at the box velocity along its x-axis
                localbox = Quaternion.rotate(Quaternion.conj( Rigid.Frame( shared.box.position[0] ).rotation ),
                                  shared.box.velocity[0][:3])
                vel = localbox[0]

                #print 'plane/ground angle:', self.currentAngle , 'mu = ', mu
                #print 'velocity:',vel

#                print vel, currentMu, shared.mu
#                print vel <= 1e-3, currentMu<shared.mu-1e-3
#                sys.stdout.flush()
                                
                EXPECT_FALSE( (vel > 1e-3) ^ (currentMu>=shared.mu-1e-2), 'mu='+str(shared.mu) ) # xor

        
        # all finished
        if currentMu >= shared.mu + .1:
            self.sendSuccess()
        
        
        # update plane orientation
        self.currentAngle += 0.001
        q = Quaternion.from_euler( [0,0,-self.currentAngle] )
        p = shared.plane.position
        p[0] = [0,0,0,q[3],q[2],q[1],q[0]]
        shared.plane.position = p
               
        return 0
          
    def bwdInitGraph(self,node):
        return 0
