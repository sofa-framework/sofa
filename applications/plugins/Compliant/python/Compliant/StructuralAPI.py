## @package StructuralAPI
# An alternative high(mid?)-level python API to describe a Compliant scene.
#
# With this API, the SOFA structure is not hidden and must be known by the user.
# But the obscure python-based structures (like Frame) are hidden.
#
# An important advantage is that a pointer is accessible for any created node and component
# (simplifying component customization, manual sub-scene creation, etc.)
#
# Note that both python APIs and manual creation can be used together.
#
# see Compliant/examples/StructuralAPI.py for a basic example.

import Frame
import Tools
from Tools import cat as concat
import numpy
import Sofa
from SofaPython import Quaternion
import SofaPython.Tools
import SofaPython.mass
import math

# to specify the floating point encoding:
# "d" to force double
# "f" to force float
# "" to let the template aliases mechanism chose according sofa compilation option (USE_FLOAT and USE_DOUBLE)
template_suffix=""

# global variable to give a different name to each visual model
idxVisualModel = 0

# to use geometric_stiffness of rigid mappings
# @warning WIP, the API will change
geometric_stiffness = 0


class RigidBody:
    ## Generic Rigid Body


    def __init__(self, node, name):
        self.node = node.createChild( name )  # node
        self.collision = None # the added collision mesh if any
        self.visual = None # the added visual model if any
        self.dofs = None   # dofs
        self.mass = None   # mass
        self.frame = Frame.Frame()
        self.framecom = Frame.Frame()
        self.fixedConstraint = None # to fix the RigidBody

    def setFromMesh(self, filepath, density = 1000, offset = [0,0,0,0,0,0,1], scale3d=[1,1,1], inertia_forces = False ):
        ## create the rigid body from a mesh (inertia and com are automatically computed)
        massInfo = SofaPython.mass.RigidMassInfo()
        massInfo.setFromMesh(filepath, density, scale3d)
        self.setFromRigidInfo(massInfo, offset, inertia_forces)

    def setFromRigidInfo(self, info, offset = [0,0,0,0,0,0,1], inertia_forces = False) :
        self.framecom = Frame.Frame()
        self.framecom.rotation = info.inertia_rotation
        self.framecom.translation = info.com

        self.frame = Frame.Frame(offset) * self.framecom

        self.dofs = self.frame.insert( self.node, name = 'dofs', template="Rigid3"+template_suffix )
        self.mass = self.node.createObject('RigidMass',
                                name = 'mass',
                                mass = info.mass,
                                inertia = concat(info.diagonal_inertia),
                                inertia_forces = inertia_forces )

    def setManually(self, offset = [0,0,0,0,0,0,1], mass = 1, inertia = [1,1,1], inertia_forces = False ):
        ## create the rigid body by manually giving its inertia
        self.frame = Frame.Frame( offset )
        self.dofs = self.frame.insert( self.node, name='dofs', template="Rigid3"+template_suffix )
        self.mass = self.node.createObject('RigidMass',
                                name = 'mass',
                                mass = mass,
                                inertia = concat(inertia),
                                inertia_forces = inertia_forces )

    def addCollisionMesh(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1], name_suffix=''):
        ## adding a collision mesh to the rigid body with a relative offset
        # (only a Triangle collision model is created, more models can be added manually)
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mechanism could be added
        self.collision = RigidBody.CollisionMesh( self.node, filepath, scale3d, ( self.framecom.inv() * Frame.Frame(offset) ).offset(), name_suffix )
        return self.collision

    def addVisualModel(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1], name_suffix=''):
        ## adding a visual model to the rigid body with a relative offset
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mechanism could be added
        self.visual = RigidBody.VisualModel( self.node, filepath, scale3d, ( self.framecom.inv() * Frame.Frame(offset) ).offset(), name_suffix )
        return self.visual

    def addOffset(self, name, offset=[0,0,0,0,0,0,1]):
        ## adding a relative offset to the rigid body (e.g. used as a joint location)
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mechanism could be added
        return RigidBody.Offset( self.node, name, ( self.framecom.inv() * Frame.Frame(offset) ).offset() )

    def addAbsoluteOffset(self, name, offset=[0,0,0,0,0,0,1]):
        ## adding a offset given in absolute coordinates to the rigid body
        return RigidBody.Offset( self.node, name, (self.frame.inv()*Frame.Frame(offset)).offset() )

    def addMappedPoint(self, name, relativePosition=[0,0,0]):
        ## adding a relative position to the rigid body
        # @warning the translation due to the center of mass offset is automatically removed. If necessary a function without this mechanism could be added
        frame = Frame.Frame(); frame.translation = relativePosition
        return RigidBody.MappedPoint( self.node, name, (self.framecom.inv()*frame).translation )

    def addAbsoluteMappedPoint(self, name, position=[0,0,0]):
        ## adding a position given in absolute coordinates to the rigid body
        frame = Frame.Frame(); frame.translation = position
        return RigidBody.MappedPoint( self.node, name, (self.frame.inv()*frame).translation )

    def addMotor( self, forces=[0,0,0,0,0,0] ):
        ## adding a constant force/torque to the rigid body (that could be driven by a controller to simulate a motor)
        return self.node.createObject('ConstantForceField', template='Rigid3'+template_suffix, name='motor', points='0', forces=concat(forces))

    def setFixed(self, isFixed=True):
        """ Add/remove a fixed constraint for this rigid, also set spedd to 0
        """
        if isFixed and self.fixedConstraint is None:
            self.fixedConstraint = self.node.createObject("FixedConstraint", name="fixedConstraint")
            self.dofs.velocity=[0,0,0,0,0,0]
            self.fixedConstraint.init()
        elif not isFixed and not self.fixedConstraint is None:
            self.node.removeObject(self.fixedConstraint)
            self.fixedConstraint=None

    class CollisionMesh:

        def __init__(self, node, filepath, scale3d, offset, name_suffix=''):
            self.node = node.createChild( "collision"+name_suffix )  # node
            r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
            self.loader = SofaPython.Tools.meshLoader(self.node, filename=filepath, name='loader', scale3d=concat(scale3d), translation=concat(offset[:3]) , rotation=concat(r), triangulate=True)
            self.topology = self.node.createObject('MeshTopology', name='topology', src="@loader" )
            self.dofs = self.node.createObject('MechanicalObject', name='dofs', template="Vec3"+template_suffix )
            self.triangles = self.node.createObject('TriangleModel', name='model')
            self.mapping = self.node.createObject('RigidMapping', name="mapping")
            self.normals = None
            self.visual = None

        def addNormals(self, invert=False):
            ## add a component to compute mesh normals at each timestep
            self.normals = self.node.createObject("NormalsFromPoints", template='Vec3'+template_suffix, name="normalsFromPoints", position='@'+self.dofs.name+'.position', triangles='@'+self.topology.name+'.triangles', quads='@'+self.topology.name+'.quads', invertNormals=invert )

        def addVisualModel(self):
            ## add a visual model identical to the collision model
            self.visual = RigidBody.CollisionMesh.VisualModel( self.node )
            return self.visual

        class VisualModel:
            def __init__(self, node ):
                global idxVisualModel;
                self.node = node.createChild( "visual" )  # node
                # todo improve normal updates by using the Rigid Transform rather than by doing cross product
                # enforcing mesh loading in VisualModel to have correct texture coordinates
                self.model = self.node.createObject('VisualModel', name="model"+str(idxVisualModel), useNormals=False, updateNormals=True, fileMesh="@../loader.filename" )
                self.mapping = self.node.createObject('IdentityMapping', name="mapping")
                idxVisualModel+=1


    class VisualModel:
        def __init__(self, node, filepath, scale3d, offset, name_suffix=''):
            global idxVisualModel;
            self.node = node.createChild( "visual"+name_suffix )  # node
            r = Quaternion.to_euler(offset[3:])  * 180.0 / math.pi
            self.model = self.node.createObject('VisualModel', name="visual"+str(idxVisualModel), fileMesh=filepath,
                                                scale3d=concat(scale3d), translation=concat(offset[:3]) , rotation=concat(r),
                                                useNormals=False, updateNormals=True)
            self.mapping = self.node.createObject('RigidMapping', name="mapping")
            idxVisualModel+=1

    class Offset:
        def __init__(self, node, name, offset):
            self.node = node.createChild( name )
            self.frame = Frame.Frame( offset ) # store the offset, relative position to its reference
            self.dofs = self.frame.insert( self.node, name='dofs', template="Rigid3"+template_suffix ) # current absolute position of this offset
            self.mapping = self.node.createObject('AssembledRigidRigidMapping', name="mapping", source = '0 '+str(self.frame), geometricStiffness=geometric_stiffness)

        def addOffset(self, name, offset=[0,0,0,0,0,0,1]):
            ## adding a relative offset to the offset
            return RigidBody.Offset( self.node, name, offset )

        def addAbsoluteOffset(self, name, offset=[0,0,0,0,0,0,1]):
            ## adding a offset given in absolute coordinates to the offset
            return RigidBody.Offset( self.node, name, (Frame.Frame(offset) * self.frame.inv()).offset() )

        def addMotor( self, forces=[0,0,0,0,0,0] ):
            ## adding a constant force/torque at the offset location (that could be driven by a controller to simulate a motor)
            return self.node.createObject('ConstantForceField', template='Rigid3'+template_suffix, name='motor', points='0', forces=concat(forces))

        def addMappedPoint(self, name, relativePosition=[0,0,0]):
            ## adding a relative position to the rigid body
            return RigidBody.MappedPoint( self.node, name, relativePosition )

        def addAbsoluteMappedPoint(self, name, position=[0,0,0]):
            ## adding a position given in absolute coordinates to the rigid body
            frame = Frame.Frame(); frame.translation = position
            return RigidBody.MappedPoint( self.node, name, (frame * self.frame.inv()).translation )

        class MappedPoint:
            def __init__(self, node, name, position):
                self.node = node.createChild( name )
                self.dofs = self.node.createObject( 'MechanicalObject', name='dofs', template="Vec3"+template_suffix, position=concat(position) )
                self.mapping = self.node.createObject('RigidMapping', name="mapping", geometricStiffness = geometric_stiffness)

    class MappedPoint:
        def __init__(self, node, name, position):
            self.node = node.createChild( name )
            self.dofs = self.node.createObject( 'MechanicalObject', name='dofs', template="Vec3"+template_suffix, position=concat(position) )
            self.mapping = self.node.createObject('RigidMapping', name="mapping", geometricStiffness = geometric_stiffness)


class GenericRigidJoint:
    ## Generic kinematic joint between two Rigids

    def __init__(self, name, node1, node2, mask, compliance=0, index1=0, index2=0):
        self.node = node1.createChild( name )
        self.mask=mask
        self.dofs = self.node.createObject('MechanicalObject', template = 'Vec6'+template_suffix, name = 'dofs', position = '0 0 0 0 0 0' )
        input = [] # @internal
        input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
        if not node2 is None:
            input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
            self.mapping = self.node.createObject('RigidJointMultiMapping', name = 'mapping', input = concat(input), output = '@dofs', pairs = str(index1)+" "+str(index2), geometricStiffness = geometric_stiffness)
            node2.addChild( self.node )
        else:
            self.mapping = self.node.createObject('RigidJointMapping', name = 'mapping', input = concat(input), output = '@dofs', pairs = str(index1)+" "+str(index2), geometricStiffness = geometric_stiffness)
        self.constraint = GenericRigidJoint.Constraint( self.node, mask, compliance )

    class Constraint:
        def __init__(self, node, mask, compliance):
            self.node = node.createChild( "constraint" )
            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs')
            self.mapping = self.node.createObject('MaskMapping',dofs=concat(mask))
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
            # self.type = self.node.createObject('Stabilization') # do not add this component, it depends on the compliance value. The assembly will automatically add the best one

    class Limits:
        def __init__(self, node, masks, limits, compliance):
            self.node = node.createChild( "limits" )

            set = []
            position = [0] * len(masks)
            offset = []

            for i in range(len(masks)):
                set = set + [0] + masks[i]
                offset.append(limits[i])

            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs', position=concat(position))
            self.mapping = self.node.createObject('ProjectionMapping', set=concat(set), offset=concat(offset))
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
            # self.type = self.node.createObject('Stabilization')
            self.constraint = self.node.createObject('UnilateralConstraint')

    def addLimits( self, limits, compliance=0 ):
            ## limits is a list of limits in each unconstrained directions
            ## always lower and upper bounds per direction, so its size is two times the number of unconstrained directions
            ## (following the order txmin,txmax,tymin,tymax,tzmin,tzmax,rxmin,rxmax,rymin,rymax,rzmin,rzmax)
            l = 0
            limitMasks=[]
            hasLimits = False
            for m in xrange(6):
                if self.mask[m] == 0 and limits[l] is not None and limits[l+1] is not None: # unconstrained direction with limits
                    hasLimits = True
                    limits[l+1] *= -1.0 # inverted upper bound
                    l += 2
                    limitMaskL = [0]*6
                    limitMaskU = [0]*6
                    # lower bound
                    limitMaskL[m] = 1
                    limitMasks.append( limitMaskL )
                    # upper bound
                    limitMaskU[m] = -1  # inverted upper bound
                    limitMasks.append( limitMaskU )
            if hasLimits:
                return GenericRigidJoint.Limits( self.node, limitMasks, limits, compliance )
            else:
                return None

    def addDamper( self, damping ):
        return self.node.createObject( 'UniformVelocityDampingForceField', dampingCoefficient=damping )

    class VelocityController:
        def __init__(self, node, mask, velocities, compliance):
            self.node = node.createChild( "controller" )
            position = [0] * len(mask)
            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs', position=concat(position))
            self.mapping = self.node.createObject('MaskMapping', dofs=concat(mask))
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance, isCompliance=1)
            self.type = self.node.createObject('VelocityConstraintValue', velocities=concat(velocities))

        def setVelocities( self, velocities ):
            self.type.velocities = concat(velocities)

    class DefaultPositionController:
        """ Set the joint position to the target
        WARNING: for angular dof position, the value must be in ]-pi,pi]
        """
        def __init__(self, node, mask, target, compliance, isCompliance=False):
            self.node = node.createChild( "controller-mask" )

            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs')
            self.node.createObject('MaskMapping', dofs=concat(mask))
            self.nodeTarget = self.node.createChild( "controller-target" )
            self.nodeTarget.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs')
            self.mapping = self.nodeTarget.createObject('DifferenceFromTargetMapping', targets=concat(target))
            self.compliance = self.nodeTarget.createObject('UniformCompliance', name='compliance', compliance=compliance, isCompliance=isCompliance)
            # self.type = self.nodeTarget.createObject('Stabilization')

        def setTarget( self, target ):
            self.mapping.targets = concat(target)

    # The PositionController can be redefined
    PositionController=DefaultPositionController

    def addGenericPositionController(self, target=None, compliance=0, mask=None, isCompliance=False):
        """ Add a controller to this joint.
        The target list must match mask list, il no target is specified, current position is used as a target.
        The mask list selects the controlled dof, if mask is None the joint natural dof are used.
        """
        if not mask is None:
            m=mask
        else:
            m = [ (1 - d) for d in self.mask ]
        if not target is None:
            t=target
        else:
            t=list()
            for i,v in enumerate(m):
                if v==1:
                    t.append(self.dofs.position[0][i])
        return GenericRigidJoint.PositionController(self.node, m, t, compliance, isCompliance)

    class ForceController:
        def __init__(self, node, mask, forces):
            self.node = node.createChild( "controller" )

            size = sum(x!=0 for x in mask)

            position = [0] * size
            points = numpy.arange(0,size,1)

            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs', position=concat(position))
            self.mapping = self.node.createObject('MaskMapping', dofs=concat(mask))
            self.force = self.node.createObject('ConstantForceField', template='Vec1'+template_suffix, forces=concat(forces), points=concat(points.tolist()) )

        def setForces( self, forces ):
            self.force.forces = concat(forces)


    class Resistance:
        def __init__(self, node, mask, threshold):
            self.node = node.createChild( "resistance" )
            position = [0] * len(mask)
            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs', position=concat(position))
            self.mapping = self.node.createObject('MaskMapping', dofs=concat(mask))
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=0, isCompliance=1)
            self.type = self.node.createObject('VelocityConstraintValue', velocities=concat([0]) )
            self.constraint = self.node.createObject('ResistanceConstraint', threshold=threshold)



class CompleteRigidJoint:
    ## A complete kinematic joint between two Rigids
    # for advanced users!

    def __init__(self, name, node1, node2, compliances=[0,0,0,0,0,0], index1=0, index2=0):
        self.node = node1.createChild( name )
        self.dofs = self.node.createObject('MechanicalObject', template='Vec6'+template_suffix, name='dofs', position='0 0 0 0 0 0' )
        input = [] # @internal
        input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
        input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
        self.mapping = self.node.createObject('RigidJointMultiMapping',  name='mapping', input=concat(input), output='@dofs', pairs=str(index1)+" "+str(index2),
                                              geometricStiffness = geometric_stiffness)
        self.constraint = CompleteRigidJoint.Constraint( self.node, compliances ) # the constraint compliance cannot be in the same branch as eventual limits...
        node2.addChild( self.node )

    class Constraint:
        def __init__(self, node, compliances):
            self.node = node.createChild( "constraint" )
            self.dofs = self.node.createObject('MechanicalObject', template='Vec6'+template_suffix, name='dofs')
            self.mapping = self.node.createObject('IdentityMapping')
            self.compliance = self.node.createObject('DiagonalCompliance', name='compliance', compliance=concat(compliances))
            # self.type = self.node.createObject('Stabilization')

    class Limits:
        def __init__(self, node, masks, limits, compliances):
            self.node = node.createChild( "limits" )

            set = []
            position = [0] * len(masks)
            offset = []

            for i in range(len(masks)):
                set = set + [0] + masks[i]
                offset.append(limits[i])

            self.dofs = self.node.createObject('MechanicalObject', template='Vec1'+template_suffix, name='dofs', position=concat(position))
            self.mapping = self.node.createObject('ProjectionMapping', set=concat(set), offset=concat(offset))
            self.compliance = self.node.createObject('DiagonalCompliance', name='compliance', compliance=concat(compliances))
            # self.type = self.node.createObject('Stabilization')
            self.constraint = self.node.createObject('UnilateralConstraint')

    def addLimits( self, masks, limits, compliances ):
            return CompleteRigidJoint.Limits( self.node, masks, limits, compliances )

    def addDamper( self, dampings=[0,0,0,0,0,0] ):
            return self.node.createObject( 'DiagonalVelocityDampingForceField', dampingCoefficients=dampings )

    def addSpring( self, compliances=[0,0,0,0,0,0] ):
        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(compliances))


class HingeRigidJoint(GenericRigidJoint):
    ## Hinge/Revolute joint around the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.mask = [1] * 6; self.mask[3+axis]=0
        GenericRigidJoint.__init__(self, name, node1, node2, self.mask, compliance, index1, index2)

    def addLimits( self, lower, upper, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.Limits( self.node, [mask,(numpy.array(mask)*-1.).tolist()], [lower,-upper], compliance )

    def addSpring( self, stiffness ):
        mask = [ (1 - d) / float(stiffness) for d in self.mask ]
        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(mask))

    def addPositionController( self, target, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.PositionController( self.node, mask, [target], compliance )

    def addVelocityController( self, velocity, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.VelocityController( self.node, mask, [velocity], compliance )

    def addForceController( self, force ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.ForceController( self.node, mask, [force] )

    def addResistance( self, threshold ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.Resistance( self.node, mask, threshold )


class SliderRigidJoint(GenericRigidJoint):
    ## Slider/Prismatic joint along the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.mask = [1] * 6; self.mask[axis]=0
        GenericRigidJoint.__init__(self, name, node1, node2, self.mask, compliance, index1, index2)

    def addLimits( self, lower, upper, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.Limits( self.node, [mask,(numpy.array(mask)*-1.).tolist()], [lower,-upper], compliance )

    def addSpring( self, stiffness ):
        mask = [ (1 - d) / float(stiffness) for d in self.mask ]
        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(mask))

    def addPositionController( self, target, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.PositionController( self.node, mask, [target], compliance )

    def addVelocityController( self, velocity, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.VelocityController( self.node, mask, [velocity], compliance )

    def addForceController( self, force ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.ForceController( self.node, mask, [force] )

    def addResistance( self, threshold ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.Resistance( self.node, mask, threshold )


class CylindricalRigidJoint(GenericRigidJoint):
    ## Cylindrical joint along and around the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis, name, node1, node2, compliance=0, index1=0, index2=0 ):
        mask = [1] * 6
        mask[axis]=0
        mask[3+axis]=0
        self.axis = axis
        GenericRigidJoint.__init__(self, name, node1, node2, mask, compliance, index1, index2)

    def addLimits( self, translation_lower, translation_upper, rotation_lower, rotation_upper, compliance=0 ):
        mask_t_l = [0]*6; mask_t_l[self.axis]=1;
        mask_t_u = [0]*6; mask_t_u[self.axis]=-1;
        mask_r_l = [0]*6; mask_r_l[3+self.axis]=1;
        mask_r_u = [0]*6; mask_r_u[3+self.axis]=-1;
        return GenericRigidJoint.Limits( self.node, [mask_t_l,mask_t_u,mask_r_l,mask_r_u], [translation_lower,-translation_upper,rotation_lower,-rotation_upper], compliance )

    def addSpring( self, translation_stiffness, rotation_stiffness ):
        mask = [0]*6; mask[self.axis]=1.0/translation_stiffness; mask[3+self.axis]=1.0/rotation_stiffness;
        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(mask))


class BallAndSocketRigidJoint(GenericRigidJoint):
    ## complete Ball and Socket / Spherical joint
    ## not the most efficient, but allowing rotation controllers

    def __init__(self, name, node1, node2, compliance=0, index1=0, index2=0 ):
        GenericRigidJoint.__init__(self, name, node1, node2, [1,1,1,0,0,0], compliance, index1, index2)

    # def addLimits( self, rotationX_lower, rotationX_upper, rotationY_lower, rotationY_upper, rotationZ_lower, rotationZ_upper, compliance=0 ):
        ## such limits make no sense for ball and socket. You could simulate it by mapping a point on one object and adding constraints/collisions on it
        # mask_x_l = [0]*6; mask_x_l[3]=1;
        # mask_x_u = [0]*6; mask_x_u[3]=-1;
        # mask_y_l = [0]*6; mask_y_l[4]=1;
        # mask_y_u = [0]*6; mask_y_u[4]=-1;
        # mask_z_l = [0]*6; mask_z_l[5]=1;
        # mask_z_u = [0]*6; mask_z_u[5]=-1;
        # return GenericRigidJoint.Limits( self.node, [mask_x_l,mask_x_u,mask_y_l,mask_y_u,mask_z_l,mask_z_u], [rotationX_lower,-rotationX_upper,rotationY_lower,-rotationY_upper,rotationZ_lower,-rotationZ_upper], compliance )

    def addSpring( self, stiffness ):
        ## only isotropic stiffness makes sense
        mask = [0, 0, 0, 1.0/stiffness, 1.0/stiffness, 1.0/stiffness ]
        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(mask))

    def addPositionController( self, axis, target, compliance=0 ):
        """ control rotation around axis (0->x, 1->y, 2->z)
        """
        mask = [0]*6
        mask[3+axis]=1
        return GenericRigidJoint.PositionController( self.node, mask, [target], compliance )


class SimpleBallAndSocketRigidJoint(GenericRigidJoint):
    ## Ball and Socket / Spherical joint, much efficient than BallAndSocketRigidJoint
    ## but that cannot offer rotation controllers, neither spring and damping (would require a identitymapping level not to mix stiffness and constraint)

    def __init__(self, name, node1, node2, compliance=0, index1=0, index2=0 ):
        # GenericRigidJoint.__init__(self, name, node1, node2, [1,1,1,0,0,0], compliance, index1, index2) # GenericRigidJoint can work but is way overkill
        self.node = node1.createChild( name )
        self.dofs = self.node.createObject('MechanicalObject', template = 'Vec3'+template_suffix, name = 'dofs', position = '0 0 0' )
        input = [] # @internal
        input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
        input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
        self.mapping = self.node.createObject('DifferenceMultiMapping', name = 'mapping', input = concat(input), output = '@dofs', pairs = str(index1)+" "+str(index2))
        self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
        node2.addChild( self.node )



class PlanarRigidJoint(GenericRigidJoint):
    ## Planar joint for the given axis as plane normal (0->x, 1->y, 2->z)

    def __init__(self, normal, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.normal = normal
        mask = [1]*6; mask[(normal+1)%3]=0; mask[(normal+2)%3]=0
        GenericRigidJoint.__init__(self, name, node1, node2, mask, compliance, index1, index2)

    def addLimits( self, translation1_lower, translation1_upper, translation2_lower, translation2_upper, compliance=0 ):
        axis1 = (self.normal+1)%3; axis2 = (self.normal+2)%3
        if axis1 > axis2 :
            axis1, axis2 = axis2, axis1
        mask_t1_l = [0]*6; mask_t1_l[axis1]=1;
        mask_t1_u = [0]*6; mask_t1_u[axis1]=-1;
        mask_t2_l = [0]*6; mask_t2_l[axis2]=1;
        mask_t2_u = [0]*6; mask_t2_u[axis2]=-1;
        return GenericRigidJoint.Limits( self.node, [mask_t1_l,mask_t1_u,mask_t2_l,mask_t2_u], [translation1_lower,-translation1_upper,translation2_lower,-translation2_upper], compliance )

    def addSpring( self, stiffness1, stiffness2 ):
        axis1 = (self.normal+1)%3; axis2 = (self.normal+2)%3
        if axis1 > axis2 :
            axis1, axis2 = axis2, axis1
        mask = [0]*6; mask[axis1]=1.0/stiffness1; mask[axis2]=1.0/stiffness2;
        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(mask))


#class GimbalRigidJoint(GenericRigidJoint):
#    ## Gimbal/Universal joint
## TODO it cannot be simulated a unique joint, but with 2 hinges with a third body in between

#    def __init__(self, axis, name, node1, node2, compliance=0, index1=0, index2=0 ):
#        self.axis = axis
#        mask = [1]*6; mask[3+(axis+1)%3]=0; mask[3+(axis+2)%3]=0
#        GenericRigidJoint.__init__(self, name, node1, node2, mask, compliance, index1, index2)

#    def addLimits( self, rotation1_lower, rotation1_upper, rotation2_lower, rotation2_upper, compliance=0 ):
#        index1 = 3+(self.axis+1)%3; index2 = 3+(self.axis+2)%3
#        if index1 > index2 :
#            index1, index2 = index2, index1
#        mask_1_l = [0]*6; mask_1_l[index1]=1;
#        mask_1_u = [0]*6; mask_1_u[index1]=-1;
#        mask_2_l = [0]*6; mask_2_l[index2]=1;
#        mask_2_u = [0]*6; mask_2_u[index2]=-1;
#        return GenericRigidJoint.Limits( self.node, [mask_1_l,mask_1_u,mask_2_l,mask_2_u], [rotation1_lower,-rotation1_upper,rotation2_lower,-rotation2_upper], compliance )

#    def addSpring( self, stiffness1, stiffness2 ):
#        index1 = 3+(self.axis+1)%3; index2 = 3+(self.axis+2)%3
#        if index1 > index2 :
#            index1, index2 = index2, index1
#        mask = [0]*6; mask[index1]=1.0/stiffness1; mask[index2]=1.0/stiffness2
#        return self.node.createObject('DiagonalCompliance', isCompliance="0", compliance=concat(mask))



class FixedRigidJoint(GenericRigidJoint):
    ## Fixed joint

    def __init__(self, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.node = node1.createChild( name )
        self.dofs = self.node.createObject('MechanicalObject', template = 'Vec6'+template_suffix, name = 'dofs', position = '0 0 0 0 0 0' )
        input = [] # @internal
        input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
        input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
        self.mapping = self.node.createObject('RigidJointMultiMapping', name = 'mapping', input = concat(input), output = '@dofs', pairs = str(index1)+" "+str(index2),
                                              geometricStiffness = geometric_stiffness)
        self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
        node2.addChild( self.node )

class DistanceRigidJoint:
    ## keep Distance between two rigid frames

    def __init__(self, name, node1, node2, compliance=0, index1=0, index2=0, rest_lenght=-1 ):
        self.node = node1.createChild( name )
        self.dofs = self.node.createObject('MechanicalObject', template='Rigid3'+template_suffix, name='dofs' )
        input = [] # @internal
        input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
        input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
        self.mapping = self.node.createObject('SubsetMultiMapping', name='mapping', input = concat(input), output = '@dofs', indexPairs="0 "+str(index1)+" 1 "+str(index2) )
        self.constraint = DistanceRigidJoint.Constraint(self.node, compliance, rest_lenght)
        node2.addChild( self.node )


    class Constraint:
        def __init__(self, node, compliance, rest_length ):
            self.node = node.createChild( 'constraint' )
            self.dofs = self.node.createObject('MechanicalObject', template = 'Vec1'+template_suffix, name = 'dofs', position = '0' )
            self.topology = self.node.createObject('EdgeSetTopologyContainer', edges="0 1" )
            self.mapping = self.node.createObject('DistanceMapping',  name='mapping', rest_length=(rest_length if rest_length>0 else "" ) )
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
            # self.type = self.node.createObject('Stabilization')

class RigidJointSpring:
    ## A 6D spring between two Rigids

    def __init__(self, name, node1, node2, stiffnesses=[0,0,0,0,0,0], index1=0, index2=0):
            self.node = node1.createChild( name )
            self.dofs = self.node.createObject('MechanicalObject', template = 'Vec6'+template_suffix, name = 'dofs', position = '0 0 0 0 0 0' )
            input = [] # @internal
            input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
            input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
            
            self.mapping = self.node.createObject('RigidJointMultiMapping',  name = 'mapping', input = concat(input), output = '@dofs', pairs = str(index1)+" "+str(index2),
                                                  geometricStiffness = geometric_stiffness)
            compliances = 1./numpy.array(stiffnesses);
            self.compliance = self.node.createObject('DiagonalCompliance', name='compliance', compliance=concat(compliances), isCompliance=0)
            node2.addChild( self.node )

## @TODO handle joints with diagonalcompliance / diagonaldamper...
## @TODO add mappings for more complex joints (eg with coupled dofs ie skrew, winch...)
## @TODO add a 6D Rigid Spring with a full stiffness matrix (not only diagonal)

