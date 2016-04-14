from __future__ import print_function

from Compliant import tool

def createScene(node):

    # make sure compliant is loaded
    compliant = node.createObject('RequiredPlugin',
                                  pluginName = 'Compliant')
    
    # some mechanical state
    src = node.createObject('MechanicalObject',
                              template = 'Rigid3d',
                              position = '1 2 3 0 0 0 1',
                              name = 'dofs')

    mass = node.createObject('RigidMass',
                             template = 'Rigid3d',
                             name = 'mass',
                             inertia = '1 1 1')

    ode = node.createObject('CompliantImplicitSolver')
    num = node.createObject('MinresSolver')

    # a rigid -> vec3 python mapping
    child = node.createChild('child')
    dst = child.createObject('MechanicalObject',
                               template = 'Vec3d',
                               position = '0 0 0',
                               name = 'dofs')

    class MyMapping(tool.Mapping):

        def __init__(self, node, **kwargs):
            tool.Mapping.__init__(self, node, **kwargs)

            # jacobian is fixed
            self.jacobian[:3, :3] = numpy.identity(3)
        
        def update(self):
            print('python mapping update')

            # map input translation
            self.value[0] = self.in_pos[0][0, :3]
            
    mapping = MyMapping(child,
                        name = 'mapping',
                        input = [src],               
                        output = dst)

    print('init done')
