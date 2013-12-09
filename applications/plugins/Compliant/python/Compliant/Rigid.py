import Sofa

from subprocess import Popen, PIPE


# small helper
def concat(x):
        return ' '.join(map(str, x))


class Frame:
        def __init__(self):
                self.translation = [0, 0, 0]
                self.rotation = [0, 0, 0, 1]
                
        def insert(self, parent, **args):
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
                return self

# TODO remove ?
def mesh_offset( mesh, path = "" ):
	str = subprocess.Popen("GenerateRigid " + mesh ).stdout.read()
        
class MassInfo:
        pass

# density is kg/m^3
def generate_rigid(filename, density = 1000.0):
        cmd = Sofa.build_dir() + '/bin/GenerateRigid'
        args = filename
	try:
		output = Popen([cmd, args], stdout=PIPE)
		line = output.stdout.read().split('\n')
	except OSError:
		print 'error when calling GenerateRigid, do you have GenerateRigid built in SOFA ?'
		raise

        start = 2
        
        # print line 
        
        mass = float( line[start].split(' ')[1] )
        volm = float( line[start + 1].split(' ')[1] )
        inrt = map(float, line[start + 2].split(' ')[1:] )
        com = map(float, line[start + 3].split(' ')[1:] )
        
        # TODO extract principal axes basis if needed
        # or at least say that we screwd up
        
        res = MassInfo()

        # by default, GenerateRigid assumes 1000 kg/m^3 already
        res.mass = (density / 1000.0) * mass

        res.inertia = [mass * x for x in inrt]
        res.com = com

        return res

# TODO provide synchronized members with sofa states ?
class Body:
        
        def __init__(self, name = "unnamed"):
                self.name = name         # node name
                self.collision = None # collision mesh
                self.visual = None    # visual mesh
                self.dofs = Frame()   # initial dofs
                self.mass = 1         # mass 
                self.inertia = [1, 1, 1] # inertia tensor
                self.color = [1, 1, 1]   # not sure this is used 
                self.offset = None       # rigid offset for com/inertia axes
                self.inertia_forces = 'false' # compute inertia forces flag
                
                # TODO more if needed (scale, color)
                
        def mass_from_mesh(self, name, density = 1000.0):
                info = generate_rigid(name, density)

                self.mass = info.mass
                
                # TODO svd inertia tensor, extract rotation quaternion
                
                self.inertia = [info.inertia[0], 
                                info.inertia[3 + 1],
                                info.inertia[6 + 2]]
                
                self.offset = Frame()
                self.offset.translation = info.com
                
                # TODO handle principal axes
                
        def insert(self, node):
                res = node.createChild( self.name )

                dofs = self.dofs.insert(res, name = 'dofs' )
                
                mass_node = res
                
                if self.offset != None:
                        mass_node = res.createChild('mapped mass')
                        self.offset.insert(mass_node, name = 'dofs')
                        mapping = mass_node.createObject('AssembledRigidRigidMapping',
                                                         template = 'Rigid',
                                                         source = '0 ' + str( self.offset) )
                # mass
                mass = mass_node.createObject('RigidMass', 
                                              template = 'Rigid',
                                              name = 'mass', 
                                              mass = self.mass, 
                                              inertia = concat(self.inertia),
                                              inertia_forces = self.inertia_forces )
                
                # visual model
                if self.visual != None:
                        visual_template = 'ExtVec3f'
                        
                        visual = res.createChild( 'visual' )
                        ogl = visual.createObject('OglModel', 
                                                  template = visual_template, 
                                                  name='mesh', 
                                                  fileMesh = self.visual, 
                                                  color = concat(self.color), 
                                                  scale3d='1 1 1')
                        
                        visual_map = visual.createObject('RigidMapping', 
                                                         template = 'Rigid' + ', ' + visual_template, 
                                                         input = '@../')
                # collision model
                if self.collision != None:
                        collision = res.createChild('collision')
                
                        collision.createObject("MeshObjLoader", name = 'loader', filename = self.collision )
                        collision.createObject('MeshTopology', 
                                               name = 'topology',
                                               triangles = '@loader.triangles')
                        collision.createObject('MechanicalObject',
                                               name = 'dofs',
                                               position = '@loader.position')
                        collision.createObject('TriangleModel', 
                                               template = 'Vec3d' )
                        collision.createObject('RigidMapping',
                                               template = 'Rigid,Vec3d',
                                               input = '@../',
                                               output = '@./')

                        # TODO lol
                
                return res


class Joint:

        def __init__(self, name = 'joint'):
                self.dofs = [0, 0, 0, 0, 0, 0]
                self.body = []
                self.offset = []
                self.name = name

                # hard constraints compliance
                self.compliance = 0
                
                # free dof stiffness/damping
                self.stiffness = 0
                self.damping = 0

                
        def append(self, node, offset = None):
                self.body.append(node)
                self.offset.append(offset)
                self.name = self.name + '-' + node.name
        
        class Node:
                pass
        
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
                
                constrained_dofs = [ (1 - d) for d in self.dofs ]

                if self.stiffness > 0:
                        # stiffness: we map all dofs
                        mask = [1, 1, 1, 1, 1, 1]
                else:
                        # only constrained dofs are needed
                        mask = constrained_dofs
                
                value = [ ( 1 - x) * self.compliance + # no dof: compliance
                          x * ( 1 / self.stiffness if self.stiffness > 0 else 0 ) # dof: 1 / stiffness or zero
                          for x in self.dofs ]
                
                map = node.createObject('RigidJointMultiMapping',
                                        name = 'mapping', 
                                        template = 'Rigid,Vec6d', 
                                        input = concat(input),
                                        output = '@dofs',
                                        dofs = concat(mask),
                                        pairs = "0 0")
                
                compliance = node.createObject('DiagonalCompliance',
                                               name = 'compliance',
                                               template = 'Vec6d',
                                               compliance = concat(value))
                
                # only stabilize constraint dofs
                stab = node.createObject('Stabilization', mask = concat( constrained_dofs ) )
                
                
                if self.damping != 0:
			# damping sub-graph
			
			dampingNode = parent.createChild(self.name + " Damping")
			
			dampingNode.createObject('MechanicalObject', 
						template = 'Vec6d', 
						name = 'dofs', 
						position = '0 0 0 0 0 0' )
						
			dampingNode.createObject('RigidJointMultiMapping',
						name = 'mapping', 
						template = 'Rigid,Vec6d', 
						input = concat(input),
						output = '@dofs',
						pairs = "0 0")
			
			dampingNode.createObject('DampingCompliance',
						 name = 'dampingCompliance',
						 template = 'Vec6d',
						 damping = self.damping)  
			
			dampingNode.createObject('DampingValue',
						 name = 'dampingValue')  
                
                
                # for some reason return node is unable to lookup for
                # children using getChild() so in the meantime...
                res = Joint.Node()
                
                res.node = node
		res.dofs = dofs
                # res.compliance = compliance
                
                
                return res

class SphericalJoint(Joint):

        def __init__(self):
                Joint.__init__(self)
                self.dofs = [0, 0, 0, 1, 1, 1]
                self.name = 'spherical-'
                



class RevoluteJoint(Joint):

        def __init__(self, axis):
                Joint.__init__(self)
                self.dofs[3 + axis] = 1
                self.name = 'revolute-'

class CylindricalJoint(Joint):

        def __init__(self, axis ):
                Joint.__init__(self)
                self.dofs[0 + axis] = 1
                self.dofs[3 + axis] = 1
                self.name = 'cylindrical-'

class PrismaticJoint(Joint):

        def __init__(self, axis):
                Joint.__init__(self)
                self.dofs[0 + axis] = 1
                self.name = 'prismatic-'

class PlanarJoint(Joint):

        def __init__(self, normal):
                Joint.__init__(self)
                self.dofs = [ 
                        int( (i != normal) if i < 3 else (i - 3 == normal) )
                        for i in xrange(6)
                ]
                self.name = 'planar-'
