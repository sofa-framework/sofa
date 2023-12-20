import Sofa

class MultiMapping(Sofa.PythonScriptController):

	def createGraph(self,node):
		
		self.rootNode = node.createChild('pythonnode')
		
		self.rootNode.createObject('EulerImplicit',name='cg_odesolver',printLog='false',rayleighStiffness='0.0', rayleighMass='0.0', vdamping='0')
		self.rootNode.createObject('CGLinearSolver',name='linear solver',iterations='100',tolerance='1.0e-10',threshold='1.0e-10')
		
		
		
		rigidNode = self.rootNode.createChild('rigid')
		rigidNode.createObject('MechanicalObject', template='Rigid', name='dofs', position="0 0 0 0 0 0 1", showObject='1', showObjectScale='0.8')
		rigidNode.createObject('UniformMass', template='Rigid', totalMass="1")
		rigidNode.createObject('PartialFixedProjectiveConstraint',indices="0",fixedDirections="1 1 1 0 0 0")

		rigidCollisionNode = rigidNode.createChild('rigid')
		rigidCollisionNode.createObject('MeshOBJLoader', filename="mesh/cube.obj", name="loader")
		rigidCollisionNode.createObject('MeshTopology', src="@loader")
		rigidCollisionNode.createObject('MechanicalObject', name="rigidCollisionDofs", template='Vec3d', position="@loader.position", showObject='1', showObjectScale='0.8')
		rigidCollisionNode.createObject('TriangleCollisionModel')
		rigidCollisionNode.createObject('RigidMapping')
		
		
		deformableNode = rigidCollisionNode.createChild('deformable')
		deformableNode.createObject('MechanicalObject', name="deformableDofs", template='Vec3d', showObject='1', showObjectScale='0.8')
		deformableNode.createObject('RegularGridTopology',min="1 -1 -1",max="3 1 1")
		deformableNode.createObject('FEMGridBehaviorModel',youngModulus="10000" ,totalMass="30" ,subdivisions="5")
		
		#deformableNode.createObject('UniformMass', template='Vec3d', totalMass="3")
		#deformableNode.createObject('HexahedronFEMForceField',youngModulus="1000",poissonRatio="0.3")
		
		
		
		deformableIndependentNode = self.rootNode.createChild('deformableIndependent')
		deformableIndependentNode.createObject('MechanicalObject', name="deformableIndependentDofs", template='Vec3d',  position="3 -1 -1  3 1 -1    3 -1 1    3 1 1", showObject='1')
		
		
		
		
		deformableNode.createObject('SubsetMultiMapping', template='Vec3d,Vec3d', input='@../../../deformableIndependent/deformableIndependentDofs @../rigidCollisionDofs', output='@deformableDofs' , indexPairs="1 0  0 0  1 4  0 1  1 1  0 2  1 5  0 3" )
		
		deformableIndependentNode.addChild( deformableNode )
		
		return 0