import Sofa

import sys
sys.path.append( Sofa.src_dir() + '/applications/plugins/Compliant/python' )
from Compliant import Rigid

colladasceneloader_path = Sofa.src_dir() + '/applications/plugins/ColladaSceneLoader'
mesh_path = colladasceneloader_path + '/Demos/'

scale = 1

def createScene(root):

    # simulation parameters
    root.dt = 0.05
    root.gravity = [0, -9.81, 0]
        
    # plugins
    root.createObject('RequiredPlugin', name='Flexible', pluginName='Flexible')
    root.createObject('RequiredPlugin', name='Compliant', pluginName='Compliant')
    root.createObject('RequiredPlugin', name='ColladaSceneLoader', pluginName='ColladaSceneLoader')
    
    # visual style
    root.createObject('VisualStyle', displayFlags='showBehavior')
    
    # scene node
    scene = root.createChild('scene')

    scene.createObject('DefaultPipeline', depth='6')
    scene.createObject('BruteForceDetection')
    scene.createObject('DefaultContactManager', responseParams='damping=0&amp;compliance=0&amp;restitution=0.5', response='CompliantContact')
    scene.createObject('NewProximityIntersection', alarmDistance='2.5', contactDistance='2.0')
    
    #scene.createObject('EulerImplicitSolver')
    #scene.createObject('CGLinearSolver', template='GraphScattered', iterations='40', tolerance='1e-009', threshold='1e-009')
    scene.createObject('AssembledSolver', stabilization='0')
    scene.createObject('SequentialSolver', precision='1e-08', relative='false', iterations='50')
    #scene.createObject('LDLTSolver') # lead to a very very big computation ... i do not know why
    #scene.createObject('MinresSolver', iterations='500')
    
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
    collisionNode.createObject('TTriangleModel', template='Vec3d')
    collisionNode.createObject('RigidMapping', template='Rigid,Vec3d', input='@../model', output='@./vertices')

    visuNode = collisionNode.createChild('visu')
    visuNode.createObject('OglModel', name='visual')
    visuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f', input='@../vertices', output='@visual')



def createChlothes(parent):

    parent.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'poncho.obj')
    parent.createObject('MechanicalObject', template='Vec3d', name='dof', position='@loader.position')
    parent.createObject('UniformMass')
    parent.createObject('MeshTopology', name='mesh', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    parent.createObject('TTriangleModel', template='Vec3d', name='models', proximity='0.2')
    parent.createObject('MeshSpringForceField', linesStiffness='1.e+4', linesDamping='0.1')
    #parent.createObject('ConstantForceField', force='9.81 -9.81 9.81', points='0') #0 24 599 623
    #parent.createObject('ConstantForceField', force='-9.81 -9.81 9.81', points='24')
    #parent.createObject('ConstantForceField', force='9.81 -9.81 -9.81', points='599')
    #parent.createObject('ConstantForceField', force='-9.81 -9.81 -9.81', points='623')
	
    createFlexibleClothes(parent)
	
    #createCompliantClothes(parent)
    
    visuNode = parent.createChild('visu')
    visuNode.createObject('OglModel', template='ExtVec3f', name='visual')
    visuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f', input='@../dof', output='@visual')
	
	
	
def createFlexibleClothes(parent):

    parent.createObject('BarycentricShapeFunction', template='ShapeFunction2d')
	
    deformationNode = parent.createChild('deformation')
    deformationNode.createObject('TopologyGaussPointSampler', name='sampler', inPosition='@../dof.position', showSamples='false', method='0', order='1')
    deformationNode.createObject('MechanicalObject', template='F321', name='triangleDeformationsDOF')
    deformationNode.createObject('LinearMapping', template="Mapping&lt;Vec3d,F321&gt;")
    
    strainNode = deformationNode.createChild('strain')
    strainNode.createObject('MechanicalObject', template='E321', name="StrainDOF")
    strainNode.createObject('CorotationalStrainMapping', template='Mapping&lt;F321,E321&gt;', method="svd") # try qr instead of svd
    strainNode.createObject('HookeForceField', template='E321', youngModulus='2000', poissonRatio='0.2', viscosity='0.1')

	

def createCompliantClothes(parent):

    extensionNode = parent.createChild('extension')
    extensionNode.createObject('MechanicalObject', template='Vec1d', name='ExtensionDOF')
    extensionNode.createObject('EdgeSetTopologyContainer', template='Vec1d', name='ExtensionEdge', edges='@../mesh.edges')
    extensionNode.createObject('ExtensionMapping', template='Vec3d,Vec1d', name="ExtensionMap", input='@../', output='@./')
    extensionNode.createObject('UniformCompliance', template='Vec1d', name="UniComp", compliance='1e-4', isCompliance='0')