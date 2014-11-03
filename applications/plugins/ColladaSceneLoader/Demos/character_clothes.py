import Sofa

from Compliant import Rigid

colladasceneloader_path = Sofa.src_dir() + '/applications/plugins/ColladaSceneLoader'
mesh_path = colladasceneloader_path + '/Demos/'

scale = 1

clothSelfCollision = 1



def createScene(root):

    # simulation parameters
    root.dt = 0.02
    root.gravity = [0, -9.81, 0]
        
    # plugins
    root.createObject('RequiredPlugin', name='Flexible', pluginName='Flexible')
    root.createObject('RequiredPlugin', name='Compliant', pluginName='Compliant')
    root.createObject('RequiredPlugin', name='ColladaSceneLoader', pluginName='ColladaSceneLoader')
    
    # visual style
    root.createObject('VisualStyle', displayFlags='showBehaviorModels')
    
    # scene node
    scene = root.createChild('scene')

    scene.createObject('DefaultPipeline', depth='6')
    scene.createObject('BruteForceDetection')
    scene.createObject('DefaultContactManager', responseParams='damping=0&amp;compliance=0&amp;restitution=0', response='CompliantContact')
    scene.createObject('MinProximityIntersection', alarmDistance='.7', contactDistance='0.5')
    
    scene.createObject('CompliantImplicitSolver', stabilization='1', warm_start=1)
    scene.createObject('SequentialSolver', precision='1e-10', iterations='100', projectH=1)
    scene.createObject('LDLTResponse', regularize=1e-18 )
    scene.createObject('CompliantAttachButtonSetting')
    
    # character (currently we use a fixed box)
    createCharacter(scene.createChild('character'))
    
    # clothes
    createChlothes(scene.createChild('clothes'))



def createCharacter(parent):

    createPersona(parent)
    #createBox(parent)

	
	
def createPersona(parent):

    parent.createObject('SceneColladaLoader', name='character', filename=mesh_path + 'lowpoly_character.dae', animationSpeed='0.14', generateCollisionModels='1')
	
	
	
def createBox(parent):

    parent.createObject('MechanicalObject', template='Rigid', name='model', position='0 100.0 0 0 0 0 1')
    parent.createObject('UniformMass')
    parent.createObject('FixedConstraint', indices='0')
    
    collisionNode = parent.createChild('collision')
    collisionNode.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'cube.obj')
    collisionNode.createObject('MeshTopology', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    collisionNode.createObject('MechanicalObject', template='Vec3d', name='vertices', position='@loader.position')
    collisionNode.createObject('Triangle', template='Vec3d')
    collisionNode.createObject('Line', template='Vec3d')
    collisionNode.createObject('Point', template='Vec3d')
    
    
    collisionNode.createObject('RigidMapping', template='Rigid,Vec3d', input='@../model', output='@./vertices')

    visuNode = collisionNode.createChild('visu')
    visuNode.createObject('OglModel', name='visual')
    visuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f', input='@../vertices', output='@visual')



def createChlothes(parent):

    parent.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'poncho.obj')
    parent.createObject('MechanicalObject', template='Vec3d', name='dof', position='@loader.position')
    parent.createObject('UniformMass')
    parent.createObject('MeshTopology', name='mesh', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    
    parent.createObject('Triangle', template='Vec3d', name='models', proximity='0', selfCollision=clothSelfCollision)
    parent.createObject('Line', template='Vec3d', name='models', proximity='0', selfCollision=clothSelfCollision)
    parent.createObject('Point', template='Vec3d', name='models', proximity='0', selfCollision=clothSelfCollision)
    
    #parent.createObject('ConstantForceField', force='9.81 -9.81 9.81', points='0') #0 24 599 623
    #parent.createObject('ConstantForceField', force='-9.81 -9.81 9.81', points='24')
    #parent.createObject('ConstantForceField', force='9.81 -9.81 -9.81', points='599')
    #parent.createObject('ConstantForceField', force='-9.81 -9.81 -9.81', points='623')
	
    
    #parent.createObject('MeshSpringForceField', linesStiffness='1.e+4', linesDamping='0.1') // spring version
	
    createFlexibleClothes(parent) # fem
	
    #createCompliantClothes(parent) # another spring version
    
    parent.createObject('FastTriangularBendingSprings', bendingStiffness=3) # bending springs
    
    visuNode = parent.createChild('visu')
    visuNode.createObject('OglModel', template='ExtVec3f', name='visual', color="red")
    visuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f', input='@../dof', output='@visual')
	
	
	
def createFlexibleClothes(parent):

    parent.createObject('BarycentricShapeFunction', template='ShapeFunctiond')
	
    deformationNode = parent.createChild('deformation')
    deformationNode.createObject('TopologyGaussPointSampler', name='sampler', inPosition='@../dof.position', showSamples='false', method='0', order='1')
    deformationNode.createObject('MechanicalObject', template='F321', name='triangleDeformationsDOF')
    deformationNode.createObject('LinearMapping', template="Vec3d,F321")
    
    strainNode = deformationNode.createChild('strain')
    strainNode.createObject('MechanicalObject', template='E321', name="StrainDOF")
    strainNode.createObject('CorotationalStrainMapping', template='F321,E321', method="svd") # try qr instead of svd
    strainNode.createObject('HookeForceField', template='E321', youngModulus='30000', poissonRatio='0.35', viscosity='0')

	

def createCompliantClothes(parent):

    extensionNode = parent.createChild('extension')
    extensionNode.createObject('MechanicalObject', template='Vec1d', name='ExtensionDOF')
    extensionNode.createObject('EdgeSetTopologyContainer', template='Vec1d', name='ExtensionEdge', edges='@../mesh.edges')
    extensionNode.createObject('ExtensionMapping', template='Vec3d,Vec1d', name="ExtensionMap", input='@../', output='@./')
    extensionNode.createObject('UniformCompliance', template='Vec1d', name="UniComp", compliance='1e-4', isCompliance='0')
    