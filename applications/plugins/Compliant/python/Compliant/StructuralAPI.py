## @package StructuralAPI
# An alternative high(mid?)-level python API to describe a Compliant scene.
#
# With this API, the SOFA structure is not hidden and must be known by the user.
# But the obscure python-based structures (like Rigid.Frame) are hidden.
#
# An important advantage is that a pointer is accessible for any created node and component
# (simplifying component customization, manual sub-scene creation, etc.)
#
# Note that both python APIs and manual creation can be used together.
#
# see Compliant/examples/StructuralAPI.py for an basic example.


import Rigid
import Tools
from Tools import cat as concat
import Vec as vec

class RigidBody:
        ## Generic Rigid Body

        def __init__(self, node, name):
                self.node = node.createChild( name )  # node
                self.dofs = 0   # dofs
                self.mass = 0   # mass
                self.frame = Rigid.Frame()
                self.framecom = Rigid.Frame() # todo handle complete frame and not only translation

        def setFromMesh(self, filepath, density = 1000.0, offset = [0,0,0,0,0,0,1], inertia_forces = False ):
                ## create the rigid body from a mesh (inertia and com are automatically computed)
                info = Rigid.generate_rigid(filepath, density)
                inertia = [info.inertia[0], info.inertia[3 + 1], info.inertia[6 + 2]]

                self.framecom = Rigid.Frame()
                self.framecom.translation = info.com

                self.frame = Rigid.Frame(offset) * self.framecom

                self.dofs = self.frame.insert( self.node, name = 'dofs' )
                self.mass = self.node.createObject('RigidMass',
                                        template = 'Rigid',
                                        name = 'mass',
                                        mass = info.mass,
                                        inertia = concat(inertia),
                                        inertia_forces = inertia_forces )

        def setManually(self, offset = [0,0,0,0,0,0,1], mass = 1, inertia = [1,1,1], inertia_forces = False ):
                ## create the rigid body by manually giving its inertia
                self.frame = Rigid.Frame( offset )
                self.dofs = self.frame.insert( self.node, name='dofs' )
                self.mass = self.node.createObject('RigidMass',
                                        template = 'Rigid',
                                        name = 'mass',
                                        mass = mass,
                                        inertia = concat(inertia),
                                        inertia_forces = inertia_forces )

        def addCollisionMesh(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1]):
            ## adding a collision mesh to the rigid body with a relative offset
            # (only a Triangle collision model is created, more models can be added manually)
            return RigidBody.CollisionMesh( self.node, filepath, scale3d, ( Rigid.Frame(offset) * self.framecom.inv() ).offset() )

        def addVisualModel(self, filepath, scale3d=[1,1,1], offset=[0,0,0,0,0,0,1]):
            ## adding a visual model to the rigid body with a relative offset
            return RigidBody.VisualModel( self.node, filepath, scale3d, ( Rigid.Frame(offset) * self.framecom.inv() ).offset() )

        def addOffset(self, name, offset=[0,0,0,0,0,0,1], index=0):
            ## adding a relative offset to the rigid body (e.g. used as a joint location)
            return RigidBody.Offset( self.node, name, ( Rigid.Frame(offset) * self.framecom.inv() ).offset(), index )

        def addAbsoluteOffset(self, name, offset=[0,0,0,0,0,0,1], index=0):
            ## adding a offset given in absolute coordinates to the rigid body
            return RigidBody.Offset( self.node, name, (Rigid.Frame(offset) * self.frame.inv()).offset(), index )

        def addMotor( self, forces=[0,0,0,0,0,0] ):
                return self.node.createObject('ConstantForceField', template='Rigid', name='motor', points='0', forces=concat(forces))

        class CollisionMesh:
            def __init__(self, node, filepath, scale3d, translation):
                    self.node = node.createChild( "collision" )  # node
                    self.loader = self.node.createObject("MeshObjLoader", name = 'loader', filename = filepath, scale3d = concat(scale3d), translation=concat(translation) )
                    self.topology = self.node.createObject('MeshTopology', name = 'topology', triangles = '@loader.triangles' )
                    self.dofs = self.node.createObject('MechanicalObject', name = 'dofs', position = '@loader.position')
                    self.triangles = self.node.createObject('TriangleModel', name = 'model', template = 'Vec3d')
                    self.mapping = self.node.createObject('RigidMapping')

        class VisualModel:
            def __init__(self, node, filepath, scale3d, translation):
                    self.node = node.createChild( "visual" )  # node
                    self.model = self.node.createObject('OglModel', template='ExtVec3f', name='model', fileMesh=filepath, scale3d=concat(scale3d), translation=concat(translation))
                    self.mapping = self.node.createObject('RigidMapping')

        class Offset:
            def __init__(self, node, name, offset, index):
                    self.node = node.createChild( name )
                    frame = Rigid.Frame( offset )
                    self.dofs = frame.insert( self.node, name='dofs' )
                    self.mapping = self.node.createObject('AssembledRigidRigidMapping', source = '0 '+str(frame))

            def addOffset(self, name, offset=[0,0,0,0,0,0,1], index=0):
                ## adding a relative offset to the offset
                return RigidBody.Offset( self.node, name, offset, index )

            def addMotor( self, forces=[0,0,0,0,0,0] ):
                    return self.node.createObject('ConstantForceField', template='Rigid', name='motor', points='0', forces=concat(forces))




class GenericRigidJoint:
    ## Generic kinematic joint between two Rigids

    def __init__(self, node, name, node1, node2, mask, compliance=0, index1=0, index2=0):
            self.node = node.createChild( name )
            self.dofs = self.node.createObject('MechanicalObject', template = 'Vec6d', name = 'dofs', position = '0 0 0 0 0 0' )
            self.input = [] # @internal
            self.input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
            self.input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
            self.mapping = self.node.createObject('RigidJointMultiMapping', template = 'Rigid,Vec6d', name = 'mapping', input = concat(self.input), output = '@dofs', pairs = str(index1)+" "+str(index2))
            self.constraint = GenericRigidJoint.Constraint( self.node, mask, compliance )

    class Constraint:
        def __init__(self, node, mask, compliance):
                self.node = node.createChild( "constraint" )
                self.dofs = self.node.createObject('MechanicalObject', template='Vec1d', name='dofs')
                self.mapping = self.node.createObject('MaskMapping',dofs=concat(mask))
                self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
                self.type = self.node.createObject('Stabilization')

    class Limits:
        def __init__(self, node, masks, limits, compliance):
            self.node = node.createChild( "limits" )

            set = []
            position = [0] * len(masks)
            offset = []

            for i in range(len(masks)):
                set = set + [0] + masks[i]
                offset.append(limits[i])

            self.dofs = self.node.createObject('MechanicalObject', template='Vec1d', name='dofs', position=concat(position))
            self.mapping = self.node.createObject('ProjectionMapping', set=concat(set), offset=concat(offset))
            self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
            self.type = self.node.createObject('ConstraintValue') # cannot be stabilized for now
            self.constraint = self.node.createObject('UnilateralConstraint')

    def addDamper( self, damping ):
            return self.node.createObject( 'UniformVelocityDampingForceField', dampingCoefficient=damping )



class CompleteRigidJoint:
    ## A complete kinematic joint between two Rigids

    def __init__(self, node, name, node1, node2, compliance=[0,0,0,0,0,0], index1=0, index2=0):
            self.node = node.createChild( name )
            self.dofs = self.node.createObject('MechanicalObject', template = 'Vec6d', name = 'dofs', position = '0 0 0 0 0 0' )
            self.input = [] # @internal
            self.input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
            self.input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
            self.mapping = self.node.createObject('RigidJointMultiMapping', template = 'Rigid,Vec6d', name = 'mapping', input = concat(self.input), output = '@dofs', pairs = str(index1)+" "+str(index2))
            self.compliance = self.node.createObject('DiagonalCompliance', name='compliance', compliance=compliance)
            self.type = self.node.createObject('Stabilization')

    class Limits:
        def __init__(self, node, masks, limits, compliances):
            self.node = node.createChild( "limits" )

            set = []
            position = [0] * len(masks)
            offset = []

            for i in range(len(masks)):
                set = set + [0] + masks[i]
                offset.append(limits[i])

            self.dofs = self.node.createObject('MechanicalObject', template='Vec1d', name='dofs', position=concat(position))
            self.mapping = self.node.createObject('ProjectionMapping', set=concat(set), offset=concat(offset))
            self.compliance = self.node.createObject('DiagonalCompliance', name='compliance', compliance=compliances)
            self.type = self.node.createObject('ConstraintValue') # cannot be stabilized for now
            self.constraint = self.node.createObject('UnilateralConstraint')

    def addLimits( self, masks, limits, compliances ):
            return CompleteRigidJoint.Limits( self.node, masks, limits, compliances )

    def addDamper( self, dampings=[0,0,0,0,0,0] ):
            return self.node.createObject( 'DiagonalVelocityDampingForceField', dampingCoefficients=dampings )


class HingeRigidJoint(GenericRigidJoint):
    ## Hinge/Revolute joint around the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis, node, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.mask = [1] * 6; self.mask[3+axis]=0
        GenericRigidJoint.__init__(self, node, name, node1, node2, self.mask, compliance, index1, index2)

    def addLimits( self, lower, upper, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.Limits( self.node, [mask,vec.minus(mask)], [lower,-upper], compliance )

    def addSpring( self, stiffness ):
        mask = [ (1 - d) for d in self.mask ]
        mask = vec.scal(1.0/stiffness,mask)
        return self.node.createObject('DiagonalCompliance', template = "Rigid", isCompliance="0", compliance=concat(mask))

class SliderRigidJoint(GenericRigidJoint):
    ## Slider/Prismatic joint along the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis, node, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.mask = [1] * 6; self.mask[axis]=0
        GenericRigidJoint.__init__(self, node, name, node1, node2, self.mask, compliance, index1, index2)

    def addLimits( self, lower, upper, compliance=0 ):
        mask = [ (1 - d) for d in self.mask ]
        return GenericRigidJoint.Limits( self.node, [mask,vec.minus(mask)], [lower,-upper], compliance )

    def addSpring( self, stiffness ):
        mask = [ (1 - d) for d in self.mask ]
        mask = vec.scal(1.0/stiffness,mask)
        return self.node.createObject('DiagonalCompliance', template = "Rigid", isCompliance="0", compliance=concat(mask))

class CylindricalRigidJoint(GenericRigidJoint):
    ## Cylindrical joint along and around the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis, node, name, node1, node2, compliance=0, index1=0, index2=0 ):
        mask = [1] * 6
        mask[axis]=0
        mask[3+axis]=0
        self.axis = axis
        GenericRigidJoint.__init__(self,node, name, node1, node2, mask, compliance, index1, index2)

    def addLimits( self, translation_lower, translation_upper, rotation_lower, rotation_upper, compliance=0 ):
        mask_t_l = [0]*6; mask_t_l[self.axis]=1;
        mask_t_u = [0]*6; mask_t_u[self.axis]=-1;
        mask_r_l = [0]*6; mask_r_l[3+self.axis]=1;
        mask_r_u = [0]*6; mask_r_u[3+self.axis]=-1;
        return GenericRigidJoint.Limits( self.node, [mask_t_l,mask_t_u,mask_r_l,mask_r_u], [translation_lower,-translation_upper,rotation_lower,-rotation_upper], compliance )

    def addSpring( self, translation_stiffness, rotation_stiffness ):
        mask = [0]*6; mask[self.axis]=1.0/translation_stiffness; mask[3+self.axis]=1.0/rotation_stiffness;
        return self.node.createObject('DiagonalCompliance', template = "Rigid", isCompliance="0", compliance=concat(mask))

class BallAndSocketRigidJoint(GenericRigidJoint):
    ## Ball and Socket / Spherical joint

    def __init__(self, node, name, node1, node2, compliance=0, index1=0, index2=0 ):
        GenericRigidJoint.__init__(self, node, name, node1, node2, [1,1,1,0,0,0], compliance, index1, index2)

    def addLimits( self, rotationX_lower, rotationX_upper, rotationY_lower, rotationY_upper, rotationZ_lower, rotationZ_upper, compliance=0 ):
        mask_x_l = [0]*6; mask_x_l[3]=1;
        mask_x_u = [0]*6; mask_x_u[3]=-1;
        mask_y_l = [0]*6; mask_y_l[4]=1;
        mask_y_u = [0]*6; mask_y_u[4]=-1;
        mask_z_l = [0]*6; mask_z_l[5]=1;
        mask_z_u = [0]*6; mask_z_u[5]=-1;
        return GenericRigidJoint.Limits( self.node, [mask_x_l,mask_x_u,mask_y_l,mask_y_u,mask_z_l,mask_z_u], [rotationX_lower,-rotationX_upper,rotationY_lower,-rotationY_upper,rotationZ_lower,-rotationZ_upper], compliance )

    def addSpring( self, stiffnessX, stiffnessY, stiffnessZ ):
        mask = [0, 0, 0, 1.0/stiffnessX, 1.0/stiffnessY, 1.0/stiffnessZ ]
        return self.node.createObject('DiagonalCompliance', template = "Rigid", isCompliance="0", compliance=concat(mask))

class PlanarRigidJoint(GenericRigidJoint):
    ## Planar joint for the given axis (0->x, 1->y, 2->z)

    def __init__(self, axis1, axis2, node, name, node1, node2, compliance=0, index1=0, index2=0 ):
        self.axis1 = axis1
        self.axis2 = axis2
        mask = [1]*6; mask[axis1]=0; mask[axis2]=0
        GenericRigidJoint.__init__(self, node, name, node1, node2, mask, compliance, index1, index2)

    def addLimits( self, translation1_lower, translation1_upper, translation2_lower, translation2_upper, compliance=0 ):
        mask_t1_l = [0]*6; mask_t1_l[self.axis1]=1;
        mask_t1_u = [0]*6; mask_t1_u[self.axis1]=-1;
        mask_t2_l = [0]*6; mask_t2_l[self.axis2]=1;
        mask_t2_u = [0]*6; mask_t2_u[self.axis2]=-1;
        return GenericRigidJoint.Limits( self.node, [mask_t1_l,mask_t1_u,mask_t2_l,mask_t2_u], [translation1_lower,-translation1_upper,translation2_lower,-translation2_upper], compliance )

    def addSpring( self, stiffness1, stiffness2 ):
        mask = [0]*6; mask[self.axis1]=1.0/stiffness1; mask[self.axis2]=1.0/stiffness2;
        return self.node.createObject('DiagonalCompliance', template = "Rigid", isCompliance="0", compliance=concat(mask))


## @TODO handle joints with diagonalcompliance / diagonaldamper...
## @TODO add mappings for more complex joint (eg with coupled dofs ie skrew, winch...)

