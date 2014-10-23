# Rigid bodies and joints, Compliant-style.
# 
# Authors: maxime.tournier@inria.fr, ... ?
#
# The basic philosophy is that python provides nice data structures to
# define scene component *semantically*. Once this is done, the actual
# scene graph must be generated through the 'insert' methods. Please
# refer to the python examples in Compliant for more information.
#

import Sofa

from subprocess import Popen, PIPE

from numpy import *
import SofaPython.Quaternion as quat
import Vec as vec

import Tools
from Tools import cat as concat

import os


class Frame:
        # a rigid frame, group operations are available.

        # TODO kwargs
        def __init__(self, value = None):
                if value != None:
                        self.translation = value[:3]
                        self.rotation = value[3:]
                else:
                        self.translation = [0, 0, 0]
                        self.rotation = [0, 0, 0, 1]
                
        def insert(self, parent, template='Rigid', **args):
                return parent.createObject('MechanicalObject', 
                                           template = template,
                                           position = str(self),
                                           **args)
                
        def __str__(self):
                return concat(self.translation) + ' ' + concat(self.rotation) 
            
        def copy(self):
                # TODO better !
                return Rigid.Frame().read( str(self) )
                                          
        def setValue(self, v):
             self.translation = v[:3]
             self.rotation = v[3:]
             return self
   
        def read(self, str):
                num = map(float, str.split())
                self.translation = num[:3]
                self.rotation = num[3:]
                return self

        def offset(self):
            return hstack((self.translation, self.rotation))

	def __mul__(self, other):
		res = Frame()
		res.translation = vec.sum(self.translation,
					  quat.rotate(self.rotation,
						      other.translation))
		res.rotation = quat.prod( self.rotation,
					  other.rotation)

		return res

	def inv(self):
		res = Frame()
		res.rotation = quat.conj( self.rotation )
		res.translation = vec.minus( quat.rotate(res.rotation,
							 self.translation) )
		return res

        def set(self, **kwargs):
                for k in kwargs:
                        setattr(self, k, kwargs[k])
                        
                return self

        # TODO more: wrench/twist frame change.

        def apply(self, vec):
            """ apply transformation to vec (a [x,y,z] vector)
            return the result
            """
            return array(quat.rotate(self.rotation, vec) + asarray(self.translation))

        def applyInv(self, vec):
            """ apply the inverse transformation to vec (a [x,y,z] vector)
            return the result
            """
            return self.inv().apply(vec)


class MassInfo:
        pass

# front-end to sofa GenerateRigid tool. density unit is kg/m^3
def generate_rigid(filename, density = 1000.0, scale=[1,1,1], rotation=[0,0,0], rigidFilename=None):

        # TODO bind GenerateRigid
        # - faster than writing in a file
        # - more robust (if several processes try to work in the same file)

        if rigidFilename is None:
            tmpfilename = Tools.path( __file__ ) +"/tmp.rigid"
        else:
            tmpfilename = rigidFilename

        cmd = [ Sofa.build_dir() + '/bin/GenerateRigid', filename, tmpfilename, str(density), str(scale[0]), str(scale[1]), str(scale[2]), str(rotation[0]), str(rotation[1]), str(rotation[2]) ]

#        print cmd
                         
        try:

            output = Popen(cmd, stdout=PIPE)

        except OSError:
            # try the debug version
            cmd[0] += 'd'

            try:
                    output = Popen(cmd, stdout=PIPE)
            except OSError:
                    print 'error when calling GenerateRigid, do you have GenerateRigid built in SOFA?'
                    raise

        output.communicate() # wait until Popen command is finished!!!
        return read_rigid(tmpfilename)

        # GenerateRigid output is stored in the file tmpfilename

# parse a .rigid file
def read_rigid(rigidFileName):
    rigidFile = open( rigidFileName, "r" )
    line = list( rigidFile )
    rigidFile.close()
#        for i in xrange(len(line)):
#            print str(i) + str(line[i])
                    
    start = 1

    res = MassInfo()

    res.mass = float( line[start].split(' ')[1] )
    #volm = float( line[start + 1].split(' ')[1])
    res.com = map(float, line[start + 3].split(' ')[1:] )

    inertia = map(float, line[start + 2].split(' ')[1:] ) # pick inertia matrix from file
    res.inertia = array( [res.mass * x for x in inertia] ).reshape( 3, 3 ) # convert it in numpy 3x3 matrix

    # extracting principal axes basis and corresponding rotation and diagonal inertia

    if inertia[1]>1e-5 or inertia[2]>1e-5 or inertia[5]>1e-5 : # if !diagonal (1e-5 seems big but the precision from a mesh is poor)
#        print res.inertia
        U, res.diagonal_inertia, V = linalg.svd(res.inertia)
        # det should be 1->rotation or -1->reflexion
        if linalg.det(U) < 0 : # reflexion
            # made it a rotation by negating a column
#            print "REFLEXION"
            U[:,0] = -U[:,0]
        res.inertia_rotation = quat.from_matrix( U )
#       print "generate_rigid not diagonal U" +str(U)
#       print "generate_rigid not diagonal V" +str(V)
#       print "generate_rigid not diagonal d" +str(res.diagonal_inertia)
    else :
        res.diagonal_inertia = res.inertia.diagonal()
        res.inertia_rotation = [0,0,0,1]

#        print "generate_rigid " + str(res.mass) + " " + str( res.inertia ) + " " + str( res.diagonal_inertia )

    return res



class Body:
        # generic rigid body
        
        def __init__(self, name = "unnamed"):
                self.name = name         # node name
                self.collision = None # collision mesh
                self.visual = None    # visual mesh
                self.dofs = Frame()   # initial dofs
                self.mass = 1         # mass 
                self.inertia = [1, 1, 1] # inertia tensor
                self.color = [1, 1, 1]   # not sure this is used 
                self.offset = None       # rigid offset for com/inertia axes
                self.inertia_forces = False # compute inertia forces flag
                self.group = None
                self.mu = 0           # friction coefficient
                self.scale = [1, 1, 1]

                # TODO more if needed (scale, color)
                
        def mass_from_mesh(self, name, density = 1000.0):
                info = generate_rigid(name, density)

                self.mass = info.mass
                
                # TODO svd inertia tensor, extract rotation quaternion
                
                self.inertia = [info.inertia[0,0],
                                info.inertia[1,1],
                                info.inertia[2,2]]
                
                self.offset = Frame()
                self.offset.translation = info.com
                
                # TODO handle principal axes
                

        def insert(self, node):
                res = node.createChild( self.name )

                # mass offset, if any
                off = Frame()
                if self.offset != None:
                        off = self.offset

                # kinematic dofs
                frame = self.dofs * off
                dofs = frame.insert(res, name = 'dofs' )
                
                # dofs are now located at the mass frame, good
                mass = res.createObject('RigidMass', 
                                        template = 'Rigid',
                                        name = 'mass', 
                                        mass = self.mass, 
                                        inertia = concat(self.inertia),
                                        inertia_forces = self.inertia_forces )
                
                # user node i.e. the one the user provided
                user = res.createChild( self.name + '-user' )
                off.inv().insert(user, name = 'dofs')
                user.createObject('AssembledRigidRigidMapping',
                                  template = 'Rigid,Rigid',
                                  source = '0 ' + str(off.inv()) )
                
                # visual model
                if self.visual != None:
                        visual_template = 'ExtVec3f'
                        
                        visual = user.createChild( 'visual' )
                        ogl = visual.createObject('OglModel', 
                                                  template = visual_template, 
                                                  name = 'mesh', 
                                                  fileMesh = self.visual, 
                                                  color = concat(self.color), 
                                                  scale3d = concat(self.scale))
                        
                        visual_map = visual.createObject('RigidMapping', 
                                                         template = 'Rigid' + ',' + visual_template,
                                                         input = '@../')
                # collision model
                if self.collision != None:
                        collision = user.createChild('collision')
                
                        collision.createObject("MeshObjLoader", 
					       name = 'loader', 
					       filename = self.collision,
                                               scale3d = concat(self.scale) )
			
                        collision.createObject('MeshTopology', 
                                               name = 'topology',
                                               triangles = '@loader.triangles')
                        
			collision.createObject('MechanicalObject',
                                               name = 'dofs',
                                               position = '@loader.position')
                        
			model = collision.createObject('TriangleModel', 
                                               name = 'model',
                                               template = 'Vec3d',
                                               contactFriction = self.mu)
                        if self.group != None:
                                model.group = self.group                        
                        
			collision.createObject('RigidMapping',
                                               template = 'Rigid,Vec3d',
                                               input = '@../',
                                               output = '@./')

                self.node = res
                self.user = user
                
                return res


class Joint:
        # generic rigid joint

        def __init__(self, name = 'joint'):
                self.dofs = [0] * 6
                self.body = []
                self.offset = []
                self.name = name
                self.pathToBodies = '' # to be used when bodies are not inserted in rootNode

                # link constraints compliance
                self.compliance = 0
                
                # TODO if you're looking for damping/stiffness, you
                # now should do it yourself, directly on joint dofs
                
        def append(self, node, offset = None):
                self.body.append(node)
                self.offset.append(offset)
                self.name = self.name + '-' + node.name
        
        # convenience: define joint using absolute frame and vararg nodes
        def absolute(self, frame, *nodes):
                for n in nodes:
                        pos = n.getObject('dofs').position
                        s = concat(pos[0])
                        local = Frame().read( s )
                        self.append(n, local.inv() * frame)
        
        # joint dimension
        def dim(self):
                return sum( self.dofs )

        class Node:
                pass
        
        def insert(self, parent):
                self.node = parent.createChild(self.name)
                
                # build input data for multimapping
                input = []
                for b, o in zip(self.body, self.offset):
                        if o is None:
                                input.append( '@' + Tools.node_path_rel(self.node, b) + '/dofs' )
                        else:
                                joint = b.createChild( self.name + '-offset' )
                                
                                joint.createObject('MechanicalObject', 
                                                   template = 'Rigid', 
                                                   name = 'dofs' )
                                
                                joint.createObject('AssembledRigidRigidMapping', 
                                                   template = "Rigid,Rigid",
                                                   source = '0 ' + str( o ) )
                                
                                input.append( '@' + Tools.node_path_rel(self.node, b) + '/' + joint.name + '/dofs' )
                             
                if len(input) == 0:
                        print 'warning: empty joint'
                        return None
   
                
                dofs = self.node.createObject('MechanicalObject', 
                                         template = 'Vec6d', 
                                         name = 'dofs', 
                                         position = '0 0 0 0 0 0' )

                map = self.node.createObject('RigidJointMultiMapping',
                                        name = 'mapping', 
                                        template = 'Rigid,Vec6d', 
                                        input = concat(input),
                                        output = '@dofs',
                                        pairs = "0 0")
                
		sub = self.node.createChild("constraints")

		sub.createObject('MechanicalObject', 
				 template = 'Vec1d', 
				 name = 'dofs')
		
		mask = [ (1 - d) for d in self.dofs ]
		
		map = sub.createObject('MaskMapping', 
				       name = 'mapping',
				       template = 'Vec6d,Vec1d',
				       input = '@../',
				       output = '@dofs',
				       dofs = concat(mask) )
		
                compliance = sub.createObject('UniformCompliance',
					      name = 'compliance',
					      template = 'Vec1d',
					      compliance = self.compliance)

                stab = sub.createObject('Stabilization')
                
		return self.node

        def setTargetPose(self, targetPose, compliance="1e-3", damping="1e3"):
            """ Set the target pose of the joint - 
            targetPose vector is filtered out according to free dofs of the joint
            """
            target = self.node.createChild("target")

            maskedTargetPose=[]
            print "dofs", self.dofs
            for i,d in enumerate(self.dofs):
                if d is 1:
                    maskedTargetPose.append(targetPose[i])
            print "maskedTargetPose", maskedTargetPose

            target.createObject('MechanicalObject', template = 'Vec1d', name = 'dofs')
            target.createObject('MaskMapping', name = 'mapping', template = 'Vec6d,Vec1d', input = '@../', output = '@dofs', dofs = "0 0 0 1 1 1" )
            
            target_constraint = target.createChild("target_constraint")
            target_constraint.createObject('MechanicalObject', template = 'Vec1d', name = 'dofs')
            target_constraint.createObject('DifferenceFromTargetMapping', name = 'mapping', template = 'Vec1d,Vec1d', input = '@../', output = '@dofs', targets = concat(maskedTargetPose) )
            target_constraint.createObject('UniformCompliance', name = 'compliance', template = 'Vec1d', compliance = compliance, damping=damping)

# and now for more specific joints:

class SphericalJoint(Joint):

        def __init__(self, **args):
                Joint.__init__(self)
                self.dofs = [0, 0, 0, 1, 1, 1]
                self.name = 'spherical'
                
                for k in args:
                        setattr(self, k, args[k])

# this one has limits \o/
class RevoluteJoint(Joint):

        # TODO make this 'x', 'y', 'z' instead
        def __init__(self, axis, **args):
                Joint.__init__(self)
                self.dofs[3 + axis] = 1
                self.name = 'revolute'
                self.lower_limit = None
                self.upper_limit = None

                for k in args:
                        setattr(self, k, args[k])


        def insert(self, parent):
                res = Joint.insert(self, parent)

                if self.lower_limit == None and self.upper_limit == None:
                        return res
                
                limit = res.createChild('limit')

                dofs = limit.createObject('MechanicalObject', template = 'Vec1d')
                map = limit.createObject('ProjectionMapping', template = 'Vec6d,Vec1d' )

                limit.createObject('UniformCompliance', template = 'Vec1d', compliance = '0' )
                limit.createObject('UnilateralConstraint');

                # don't stabilize as we need to detect violated
                # constraints first
                # limit.createObject('Stabilization');

                set = []
                position = []
                offset = []

                if self.lower_limit != None:
                        set = set + [0] + self.dofs
                        position.append(0)
                        offset.append(self.lower_limit)

                if self.upper_limit != None:
                        set = set + [0] + vec.minus(self.dofs)
                        position.append(0)
                        offset.append(- self.upper_limit)
                
                map.set = concat(set)
                map.offset = concat(offset)
                dofs.position = concat(position)

                return res


class CylindricalJoint(Joint):

        def __init__(self, axis ):
                Joint.__init__(self)
                self.dofs[0 + axis] = 1
                self.dofs[3 + axis] = 1
                self.name = 'cylindrical'

class PrismaticJoint(Joint):

        def __init__(self, axis):
                Joint.__init__(self)
                self.dofs[0 + axis] = 1
                self.name = 'prismatic'

class PlanarJoint(Joint):

        def __init__(self, normal):
                Joint.__init__(self)
                self.dofs = [ 
                        int( (i != normal) if i < 3 else (i - 3 == normal) )
                        for i in xrange(6)
                ]
                self.name = 'planar'
