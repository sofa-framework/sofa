import Sofa

def createScene(node):

	# a generic implementation of a cheap but clean attachment between a rigid and a deformable using a SubsetMultiMapping
	# this has been called 'hard-bindings' in Sifakis, E., Shinar, T., Irving, G. and Fedkiw, R. "Hybrid simulation of deformable solids", SCA 2007

	node.gravity = [0,-2,0]

	node.createObject('VisualStyle',displayFlags="showBehaviorModels hideCollisionModels hideMappings showForceFields hideVisualModels" )


	node.createObject('EulerImplicit',name='cg_odesolver',printLog='false',rayleighStiffness='0.2', rayleighMass='0.2', vdamping='0')
	node.createObject('CGLinearSolver',name='linear solver',iterations='100',tolerance='1.0e-10',threshold='1.0e-10')



	rigidNode = node.createChild('rigid')
	rigidNode.createObject('MechanicalObject', template='Rigid', name='dofs', position="0 0 0 0 0 0 1", showObject='1', showObjectScale='0.8')

	deformableMappedNode = rigidNode.createChild('mapped_deformable_nodes_yellow')
	mappedDofs = deformableMappedNode.createObject('MechanicalObject', name="mappedDofs", template='Vec3d', position="1 -1 -1   1 1 -1  1 -1 1   1 1 1  -1 -1 -1   -1 1 -1  -1 -1 1   -1 1 1", showObject='1', showObjectScale='10', showColor="1 1 0 1")
	deformableMappedNode.createObject('RigidMapping')

	deformableIndependentNode = node.createChild('independent_deformable_nodes_white')
	independentDofs = deformableIndependentNode.createObject('MechanicalObject', name="deformableIndependentDofs", template='Vec3d',  position="3 -1 -1  3 1 -1    3 -1 1    3 1 1", showObject='1', showObjectScale='10', showColor="1 1 1 1")
	deformableIndependentNode.createObject('FixedConstraint',indices="0",drawSize=".1")



	deformableNode = deformableMappedNode.createChild('all_deformable_nodes')
	deformableNode.createObject('RegularGridTopology',min="1 -1 -1",max="3 1 1")
	deformableNode.createObject('MechanicalObject', name="deformableDofs", template='Vec3d', showObject='1', showObjectScale='0.8')
	deformableNode.createObject('UniformMass', template='Vec3d', totalMass="3")
	deformableNode.createObject('HexahedronFEMForceField',youngModulus="10",poissonRatio="0.1")
	deformableNode.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', input=independentDofs.getLinkPath()+" "+mappedDofs.getLinkPath(), output='@./' , indexPairs="1 0  0 0  1 1  0 1  1 2  0 2  1 3  0 3" )
	deformableIndependentNode.addChild( deformableNode )

	return 0