import Sofa

# utility python module to dynamically create objects in Sofa




def createCube(parentNode,name,x,y,z,vx,vy,vz,color):
	node = parentNode.createChild(name)

	node.createObject('EulerImplicit')
	node.createObject('CGLinearSolver', iterations=25, tolerance=1.0e-9, threshold=1.0e-9)

	object = node.createObject('MechanicalObject',name='MecaObject',template='Rigid', dx=x, dy=y, dz=z, vy=100, velocity=str(vx)+' '+str(vy)+' '+str(vz)+' 0 0 0')

	node.createObject('UniformMass', totalmass=100)

	# VisualNode
	VisuNode = node.createChild('Visu')
	VisuNode.createObject("OglModel", name='Visual', fileMesh='mesh/PokeCube.obj', color=color, dx=x, dy=y, dz=z)
	#desc.setAttribute('object1','@..')
	#desc.setAttribute('object2','@Visual')
	VisuNode.createObject('RigidMapping', input='@..', output='@Visual')
	

	return node

