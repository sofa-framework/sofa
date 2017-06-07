from __future__ import print_function

from SofaPython import SofaNumpy, script
from Compliant import easy, tool

import math
import numpy as np

from Compliant import mapping, easy


def particle(node, position):
    '''construct a particle under given node'''
    dofs = node.createObject('MechanicalObject', position = tool.cat(position),
                             showObject = True, drawMode = 1, name = 'dofs', size = 1)
    mass = node.createObject('UniformMass', name = 'mass', totalMass = 1)
    return dofs



class Solver(easy.Solver):

    def factor(self, sys):
        assert (sys.H == np.identity(6)).all(), "H error"
        assert (sys.J == np.array([-1, 0, 0, 1, 0, 0])).all(), "J error"
        assert (sys.C == np.identity(1)).all(), "C error"
        assert (sys.P == np.diag([0, 0, 0, 1, 1, 1])).all(), "P error"
        
    def solve(self, res, sys, rhs):
        # wtf rhs is already projected?
        assert (rhs == np.array([0, 0, 0,
                                 0, 1, 0,
                                 0])).all(), "dynamics rhs error"
        
    def correct(self, res, sys, rhs, damping):
        assert (rhs == np.zeros(7)).all(), "correction rhs error"
        


class Script(script.Controller):

    def __init__(self, node):
        self.node = node
        
    def onEndAnimationStep(self, dt):
        self.node.active = False

        
def createScene(node):
    node.createObject('CompliantImplicitSolver')

    solver = Solver(node)
    
    n1 = node.createChild('p1')

    d1 = particle(n1, [-0.5, 0, 0])
    n1.createObject("FixedConstraint", fixAll = True)
        
    n2 = node.createChild('p2')
    d2 = particle(n2, [0.5, 0, 0])
    
    n3 = node.createChild('child')
    d3 = n3.createObject('MechanicalObject', template = 'Vec1', name = 'dofs', size = 1)

    # our mapping is here
    dist = mapping.DistanceMapping(n3, input = [d1, d2], output = d3)
    dist.rest_length = 1

    # put a forcefield on mapping output
    ff = n3.createObject('UniformCompliance', template = 'Vec1',
                         isCompliance = True, compliance = 1)
    
    node.dt = 1
    node.gravity = '0 1 0'

    script = Script(node)
