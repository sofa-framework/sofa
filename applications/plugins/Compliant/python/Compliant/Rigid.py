import Sofa

# small helper
def concat(x):
        return ' '.join(map(str, x))


class Frame:
        def __init__(self):
                self.translation = [0, 0, 0]
                self.rotation = [0, 0, 0, 1]
                
        def mstate(self, parent, **args):
                return parent.createObject('MechanicalObject', 
                                           template = 'Rigid',
                                           translation = concat(self.translation),
                                           rotation = concat(self.rotation),
                                           **args)
                
        def __str__(self):
                return concat(self.translation) + ' ' + concat(self.rotation) 
               
        def read(self, str):
                num = map(float, str.split())
                self.translation = num[:3]
                self.rotation = num[3:]
        
# TODO provide synchronized members with sofa states
class Body:
        
        def __init__(self):
                self.name = "unnamed"
                self.collision = "" # collision mesh
                self.visual = ""    # visual mesh
                self.dofs = Frame()
                self.mass = 1
                self.inertia = [1, 1, 1] 
                self.template = 'Rigid'
                self.color = [1, 1, 1]
                # TODO more if needed (scale, color)
                
        def insert(self, node):
                res = node.createChild( self.name )

                dofs = self.dofs.mstate(res, name = 'dofs' )
                
                mass = res.createObject('RigidMass', 
                                        template = self.template, 
                                        name = 'mass', 
                                        mass = self.mass, 
                                        inertia = concat(self.inertia))
                
                # visual
                visual_template = 'ExtVec3f'
                
                visual = res.createChild( 'visual' )
                ogl = visual.createObject('OglModel', 
                                          template = visual_template, 
                                          name='mesh', 
                                          fileMesh = self.visual, 
                                          color = concat(self.color), 
                                          scale3d='1 1 1')
                
                visual_map = visual.createObject('RigidMapping', 
                                                 template = self.template + ', ' + visual_template, 
                                                 input = '@../')
                
                # collision
                collision = res.createChild('collision')
                
                # TODO lol
                
                return res






class Joint:

        def __init__(self):
                self.dofs = [1, 1, 1, 1, 1, 1]
                self.body = []
                self.offset = []
                self.name = 'joint'
                
        def append(self, node, offset = None):
                self.body.append(node)
                self.offset.append(offset)
                self.name = self.name + '-' + node.name
        
        def insert(self, parent):
                # build input data for multimapping
                input = []
                for b, o in zip(self.body, self.offset):
                        if o is None:
                                input.append( '@' + b.name + '/dofs' )
                        else:
                                joint = b.createChild( self.name + '-offset' )
                                
                                joint.createObject('MechanicalObject', 
                                                   template = 'Rigid', 
                                                   name = 'dofs' )
                                
                                joint.createObject('AssembledRigidRigidMapping', 
                                                   template = "Rigid,Rigid",
                                                   source = '0 ' + str( o ) )
                                
                                input.append( '@' + b.name + '/' + joint.name + '/dofs' )
                                
                # now for the joint dofs
                node = parent.createChild(self.name)
                
                dofs = node.createObject('MechanicalObject', 
                                         template = 'Vec6d', 
                                         name = 'dofs', 
                                         position = '0 0 0 0 0 0' )
                
                mask = [ 1 - d for d in self.dofs ]
                
                map = node.createObject('RigidJointMultiMapping',
                                        name = 'mapping', 
                                        template = 'Rigid,Vec6d', 
                                        input = concat(input),
                                        output = '@dofs',
                                        dofs = concat(mask),
                                        pairs = "0 0")
                
                compl = node.createObject('UniformCompliance',
                                          template = 'Vec6d',
                                          compliance = '0')
                
                stab = node.createObject('Stabilization')

class SphericalJoint(Joint):

        def __init__(self):
                Joint.__init__(self)
                self.dofs = [0, 0, 0, 1, 1, 1]
                self.name = 'spherical-joint'
                

# along x axis
class RevoluteJoint(Joint):

        def __init__(self):
                Joint.__init__(self)
                self.dofs = [0, 0, 0, 1, 0, 0]
                self.name = 'revolute-joint'

# along x axis
class CylindricalJoint(Joint):

        def __init__(self):
                Joint.__init__(self)
                self.dofs = [1, 0, 0, 1, 0, 0]
                self.name = 'cylindrical-joint'
