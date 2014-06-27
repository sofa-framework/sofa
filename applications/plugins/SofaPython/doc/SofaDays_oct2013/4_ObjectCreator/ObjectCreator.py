import Sofa

# utility python module to dynamically create objects in Sofa


def createDragon(parentNode,name,x,y,z,color):
	node = parentNode.createChild(name)
#   <EulerImplicit name="cg_odesolver" printLog="false" />
	node.createObject('EulerImplicit',name='cg_odesolver')
#   <CGLinearSolver iterations="25" name="linear solver" tolerance="1.0e-9" threshold="1.0e-9" />
	node.createObject('CGLinearSolver', name='linear solver', iterations=25, tolerance=1.0e-9, threshold=1.0e-9)

	node.createObject('MechanicalObject', dx=x, dy=y, dz=z)

	node.createObject('UniformMass',name='mass',totalmass=10)

	node.createObject('RegularGrid', name='RegularGrid1',nx=6, ny=5, nz=3, xmin=-11, xmax=11, ymin=-7, ymax=7, zmin=-4, zmax=4)
	node.createObject('RegularGridSpringForceField', name='Springs', stiffness=350, damping=1 )

	VisuNode = node.createChild('VisuDragon')
	VisuNode.createObject('OglModel', name='Visual', filename='mesh/dragon.obj', color=color, dx=x, dy=y, dz=z)
	VisuNode.createObject('BarycentricMapping', input='@..', output='@Visual')

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MeshObjLoader', name="loader", filename="mesh/dragon.obj")
	SurfNode.createObject('Mesh', src="@loader")
	SurfNode.createObject('MechanicalObject', src="@loader", dx=x, dy=y, dz=z)
	SurfNode.createObject('Triangle')
	SurfNode.createObject('Line')
	SurfNode.createObject('Point')
	SurfNode.createObject('BarycentricMapping')

	return node






def createArmadillo(parentNode,name,x,y,z,color):
	node = parentNode.createChild(name)
#   <EulerImplicit name="cg_odesolver" printLog="false" />
	node.createObject('EulerImplicit',name='cg_odesolver')
#   <CGLinearSolver iterations="25" name="linear solver" tolerance="1.0e-9" threshold="1.0e-9" />
	node.createObject('CGLinearSolver', name='linear solver', iterations=25, tolerance=1.0e-9, threshold=1.0e-9)

	node.createObject('MechanicalObject', dx=x, dy=y, dz=z)

	node.createObject('UniformMass',name='mass',totalmass=10)

#	node.createObject('SparseGridTopology', n="4 4 4", fileTopology="mesh/Armadillo_verysimplified.obj")
#	node.createObject('HexahedronFEMForceField', youngModulus=100)
	node.createObject('RegularGrid', name='RegularGrid1',nx=4, ny=4, nz=4, xmin=-6, xmax=6, ymin=-6, ymax=9, zmin=-5, zmax=5)
	node.createObject('RegularGridSpringForceField', name='Springs', stiffness=350, damping=1 )
	
	VisuNode = node.createChild('VisuArmadillo')
	VisuNode.createObject('OglModel', name='Visual', filename='mesh/Armadillo_verysimplified.obj', color=color, dx=x, dy=y, dz=z)
	VisuNode.createObject('BarycentricMapping', input='@..', output='@Visual')

	SurfNode = node.createChild('Surf')
	SurfNode.createObject('MeshObjLoader', name="loader", filename="mesh/Armadillo_verysimplified.obj")
	SurfNode.createObject('Mesh', src="@loader")
	SurfNode.createObject('MechanicalObject', src="@loader", dx=x, dy=y, dz=z)
	SurfNode.createObject('Triangle')
	SurfNode.createObject('Line')
	SurfNode.createObject('Point')
	SurfNode.createObject('BarycentricMapping')

	return node



