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


class RigidBody:
        ## Generic Rigid Body

        def __init__(self, node, name):
                self.node = node.createChild( name )  # node
                self.dofs = 0   # dofs
                self.mass = 0   # mass

        def setFromMesh(self, filepath, density = 1000.0, inertia_forces = False ):
                ## create the rigid body from a mesh (inertia and com are automatically computed)
                info = Rigid.generate_rigid(filepath, density)
                inertia = [info.inertia[0], info.inertia[3 + 1], info.inertia[6 + 2]]

                frame = Rigid.Frame()
                frame.translation = info.com

                self.dofs = frame.insert( self.node, name = 'dofs' )
                self.mass = self.node.createObject('RigidMass',
                                        template = 'Rigid',
                                        name = 'mass',
                                        mass = info.mass,
                                        inertia = concat(inertia),
                                        inertia_forces = inertia_forces )

        def setManually(self, offset = [0,0,0,0,0,0,1], mass = 1, inertia = [1,1,1], inertia_forces = False ):
                ## create the rigid body by manually giving its inertia
                frame = Rigid.Frame( offset )
                self.dofs = frame.insert( self.node, name='dofs' )
                self.mass = self.node.createObject('RigidMass',
                                        template = 'Rigid',
                                        name = 'mass',
                                        mass = mass,
                                        inertia = concat(inertia),
                                        inertia_forces = inertia_forces )

        def addCollisionMesh(self, filepath, scale3d=[1,1,1]):
            ## adding a collision mesh to the rigid body
            # (only a Triangle collision model is created, more models can be added manually)

            return RigidBody.CollisionMesh( self.node, filepath, scale3d )

        def addVisualModel(self, filepath, scale3d=[1,1,1]):
            ## adding a visual model to the rigid body
            return RigidBody.VisualModel( self.node, filepath, scale3d )

        def addOffset(self, name, offset=[0,0,0,0,0,0,1], index=0):
            ## adding an offset rgid frame the rigid body (e.g. used as a joint location)
            return RigidBody.Offset( self.node, name, offset, index )



        class CollisionMesh:
            def __init__(self, node, filepath, scale3d):
                    self.node = node.createChild( "collision" )  # node
                    self.loader = self.node.createObject("MeshObjLoader", name = 'loader', filename = filepath, scale3d = concat(scale3d) )
                    self.topology = self.node.createObject('MeshTopology', name = 'topology', triangles = '@loader.triangles' )
                    self.dofs = self.node.createObject('MechanicalObject', name = 'dofs', position = '@loader.position')
                    self.triangles = self.node.createObject('TriangleModel', name = 'model', template = 'Vec3d')
                    self.mapping = self.node.createObject('RigidMapping')

        class VisualModel:
            def __init__(self, node, filepath, scale3d):
                    self.node = node.createChild( "visual" )  # node
                    self.model = self.node.createObject('OglModel', template='ExtVec3f', name='model', fileMesh=filepath, scale3d=concat(scale3d))
                    self.mapping = self.node.createObject('RigidMapping')

        class Offset:
            def __init__(self, node, name, offset, index):
                    self.node = node.createChild( name )
                    frame = Rigid.Frame( offset )
                    self.dofs = frame.insert( self.node, name='dofs' )
                    self.mapping = self.node.createObject('AssembledRigidRigidMapping', source = '0 '+str(frame))




class GenericRigidJoint:
    ## Generic kinematic joint between two Rigids

    def __init__(self, node, name, node1, node2, mask, compliance=0, index1=0, index2=0):
            self.node = node.createChild( name )
            self.dofs = self.node.createObject('MechanicalObject', template = 'Vec6d', name = 'dofs', position = '0 0 0 0 0 0' )
            input = []
            input.append( '@' + Tools.node_path_rel(self.node,node1) + '/dofs' )
            input.append( '@' + Tools.node_path_rel(self.node,node2) + '/dofs' )
            self.mapping = self.node.createObject('RigidJointMultiMapping', template = 'Rigid,Vec6d', name = 'mapping', input = concat(input), output = '@dofs', pairs = str(index1)+" "+str(index2))
            self.constraint = GenericRigidJoint.Constraint( self.node, mask, compliance )

    class Constraint:
        def __init__(self, node, mask, compliance):
                self.node = node.createChild( "constraint" )
                self.dofs = self.node.createObject('MechanicalObject', template='Vec1d', name='dofs')
                self.mapping = self.node.createObject('MaskMapping',dofs=concat(mask))
                self.compliance = self.node.createObject('UniformCompliance', name='compliance', compliance=compliance)
                self.type = self.node.createObject('Stabilization')

## @TODO add specific kinematic joints (hinge, ball&socket...)
# note that for joint only based on translation (no rotation constraints), a differencemapping should be enough (rather than a RigidJointMultiMapping)
# and the mask mapping could be done before, leading in more efficient joint
## @TODO handle joint with diagonalcompliance
