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

    node.dt = 0.1
    
    dist = node.createChild('dist')

    d = dist.createObject('MechanicalObject',
                          template = 'Vec1d', position = '0',
                          name = 'dofs')


    class DistanceMapping(tool.Mapping):

        def __init__(self, node, **kwargs):
            tool.Mapping.__init__(self, node, **kwargs)

            # some checks
            assert len(self.in_pos) == 2, 'should have 2 input dofs'
            assert self.out_vel.shape == (1, 1), 'ouput dofs should have size 1'

            p1 = self.in_pos[0]
            p2 = self.in_pos[1]

            delta = p2 - p1
            theta = np.linalg.norm(delta)

            # init distance is rest length
            self.rest = theta

            
        def on_stiffness(self, out_force):
            '''update mapping geometric stiffness'''

            print('update gs')
            
            p1 = self.in_pos[0][0]
            p2 = self.in_pos[1][0]

            delta = p2 - p1

            theta = np.linalg.norm(delta)

            f = out_force[0][0]
            
            if theta > 0:

                # unit vector
                u = delta / theta

                # watch out: numpy takes u.dot(u.T) for the inner
                # product. use np.outer instead
                block = (np.identity(3) - np.outer(u, u)) * (f / theta)
                         
                # diagonal
                self.geometric_stiffness[:3, :3] = block
                self.geometric_stiffness[3:, 3:] = block

                # off-diagonal
                # TODO maybe only the upper/lower is needed ? 
                self.geometric_stiffness[:3, 3:] = -block
                self.geometric_stiffness[3:, :3] = -block

            else:
                self.geometric_stiffness[:] = np.zeros( (6, 6) )
                

                
        def on_apply(self):
            '''update mapping jacobian/value'''
            
            print('update')
            
            p1 = self.in_pos[0][0]
            p2 = self.in_pos[1][0]

            delta = p2 - p1
            theta = np.linalg.norm(delta)

            self.value[:] = theta - self.rest

            if theta > 0:
                # unit vector
                u = delta / theta

                self.jacobian[0, :3] = -u
                self.jacobian[0, 3:] = u                

            else:
                self.jacobian = np.zeros( (1, 6) )

    mapping = DistanceMapping(dist, input = [d1, d2], output = d)

    ff = dist.createObject('UniformCompliance', template = 'Vec1d',
                           isCompliance = True, compliance = 1e-5)
    
    
    

    
