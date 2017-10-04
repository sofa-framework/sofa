from __future__ import print_function

from SofaPython import SofaNumpy 
from Compliant import easy, tool

import math
import numpy as np


class DistanceMapping(easy.Mapping):
    '''a hooke-like distance mapping: 
    
    f(p1, p2) = norm(p2 - p1) - rest_length
    '''
    
    def __init__(self, node, **kwargs):
        easy.Mapping.__init__(self, node, **kwargs)

        self.rest_length = 0

    def apply(self, out, at):
        
        p1 = at[0][0]
        p2 = at[1][0]
        
        delta = p2 - p1
        norm = math.sqrt(delta.dot(delta))
        
        out[0][0] = norm - self.rest_length

        
    def jacobian(self, js, at):

        p1 = at[0][0]
        p2 = at[1][0]
        
        delta = p2 - p1
        norm = math.sqrt(delta.dot(delta))

        if norm > 1e-14:
            u = delta / norm
            js[0][0] = -u
            js[1][0] = u
        else:
            # warning
            js[0][0] = np.zeros(3)
            js[1][0] = np.zeros(3)


    def geometric_stiffness(self, gs, at, force):
        
        p1 = at[0][0]
        p2 = at[1][0]
        
        delta = p2 - p1

        norm = math.sqrt(delta.dot(delta))
        if norm > 1e-14:
            
            u = delta / norm
            f = force[0][0]

            block = (np.identity(3) - np.outer(u, u)) * (f / norm)
            gs[:3, :3] = block
            gs[3:, 3:] = block

            gs[:3, 3:] = -block
            gs[3:, :3] = -block

            
        else:
            gs[:, :] = 0



def particle(node, position):
    '''construct a particle under given node'''
    dofs = node.createObject('MechanicalObject', position = tool.cat(position),
                             showObject = True, drawMode = 1, name = 'dofs')
    mass = node.createObject('UniformMass', name = 'mass')
    return dofs

    
        
def createScene(node):
    node.createObject('RequiredPlugin', pluginName = 'Compliant')
    
    node.createObject('EulerImplicitSolver')
    node.createObject('CGLinearSolver')

    # node.createObject('CompliantImplicitSolver')
    # node.createObject('LDLTSolver')
    
    n1 = node.createChild('p1')

    
    d1 = particle(n1, [-0.5, 0, 0])
    n1.createObject("FixedConstraint", fixAll = True)
        
    n2 = node.createChild('p2')
    d2 = particle(n2, [0.5, 0, 0])
    
    n3 = node.createChild('child')
    d3 = n3.createObject('MechanicalObject', template = 'Vec1', name = 'dofs')

    # our mapping is here
    dist = DistanceMapping(n3,
                           input = tool.multi_mapping_input(n3, d1, d2),
                           output = '@dofs')
    dist.rest_length = 1

    # put a forcefield on mapping output
    ff = n3.createObject('UniformCompliance', template = 'Vec1',
                         isCompliance = False, compliance = 1e-5)
    
