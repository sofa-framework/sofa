from __future__ import print_function

from Compliant import tool

import numpy as np

def createScene(node):

    # make sure compliant is loaded
    compliant = node.createObject('RequiredPlugin',
                                  pluginName = 'Compliant')

    # solver
    ode = node.createObject('CompliantImplicitSolver')
    num = node.createObject('MinresSolver', iterations = 100, precision = 1e-14)

    ode.stabilization = '0'
    ode.neglecting_compliance_forces_in_geometric_stiffness = False
    # ode.propagate_lambdas = True
    
    # some mechanical state
    n1 = node.createChild('p1')
    d1 = n1.createObject('MechanicalObject',
                           template = 'Vec3d',
                           position = '-1 0 0',
                           name = 'dofs')

    m1 = n1.createObject('UniformMass', mass = 1)

    n2 = node.createChild('p2')
    d2 = n2.createObject('MechanicalObject',
                           template = 'Vec3d',
                           position = '1 0 0',
                           name = 'dofs')

    m2 = n2.createObject('UniformMass', mass = 1)


    # fix n1
    n1.createObject("FixedConstraint", indices = '0')

    d1.showObject = 1
    d1.drawMode = 1
    d1.showObjectScale = 0.1
    
    d2.showObject = 1
    d2.drawMode = 1
    d2.showObjectScale = 0.1

    node.dt = 0.01
    
    dist = node.createChild('interaction')


    class MyForceField(tool.ForceField):

        def on_force(self):
            
            p1 = self.in_pos[0][0]
            p2 = self.in_pos[1][0]

            delta = p2 - p1
            
            f = -self.k * delta

            self.force[:3] = f
            self.force[3:] = -f

            # optional
            self.energy = 0.5 * self.k * delta.dot(delta)

        def on_stiffness(self, factor):

            diag3 = np.diag_indices( 3 )
            diag6 = np.diag_indices( 6 )            

            # diagonal
            self.stiffness[ diag6 ] = - factor * self.k

            # off-diagonal
            self.stiffness[ 3:, :3 ][diag3] = factor * self.k
            self.stiffness[ :3, 3: ][diag3] = factor * self.k            
            
            
    ff = MyForceField(dist, input = [d1, d2])
    ff.k = 1e2
    
