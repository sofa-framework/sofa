from __future__ import print_function, absolute_import

import Sofa

from SofaTest import gtest
from Compliant import StructuralAPI as api
from Compliant.types import *

import sys



class Script(api.Script):

    def onEndAnimationStep(self, dt):
        axis, angle = self.rigid.position.orient.axis_angle()

        error = norm(self.expected_angle - angle)

        gtest.assert_true(error < 1e-10, 'quaternion integration error')
        gtest.finish()
    

def createScene(node):
    ode = node.createObject('CompliantImplicitSolver')
    num = node.createObject('LDLTSolver')
    
    rigid = api.RigidBody(node, 'rigid')
    rigid.setManually()

    script = Script(node)
    script.rigid = rigid

    # TODO randomize these
    node.dt = 1
    omega = 2
    
    rigid.velocity.angular[2] = omega
    script.expected_angle = node.dt * omega
    script.node = node
    
    return node


 

