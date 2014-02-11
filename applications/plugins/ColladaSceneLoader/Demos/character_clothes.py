import Sofa

# TODO standardize plugins python directory, then use something like
# Sofa.plugin_path('Compliant')

import sys
sys.path.append( Sofa.src_dir() + '/applications/plugins/Compliant/python' )

from Compliant import Rigid

colladasceneloader_path = Sofa.src_dir() + '/applications/plugins/ColladaSceneLoader'
mesh_path = colladasceneloader_path + '/Demos/'

scale = 1

def createScene(root):

    # simulation parameters
    root.dt = 1e-2
    root.gravity = [0, -9.8, 0]
        
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
    #scene.createObject('LDLTSolver')
    #scene.createObject('MinresSolver', iterations='500')
    scene.createObject('SequentialSolver', precision='1e-08', relative='false', iterations='50')
    
    # character (currently we use a fixed box)
    characterMainNode = scene.createChild('character')
    
    characterMainNode.createObject('MechanicalObject', template='Rigid', name='model', position='0 100.0 0 0 0 0 1')
    characterMainNode.createObject('RigidMass', mass='1', inertia='1 1 1')
    characterMainNode.createObject('FixedConstraint', indices='0')
    
    characterCollisionNode = characterMainNode.createChild('collision')
    characterCollisionNode.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'cube.obj')
    characterCollisionNode.createObject('MeshTopology', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    characterCollisionNode.createObject('MechanicalObject', template='Vec3d', name='vertices', position='@loader.position')
    characterCollisionNode.createObject('TTriangleModel', template='Vec3d')
    characterCollisionNode.createObject('RigidMapping', template='Rigid,Vec3d', input='@../model', output='@./vertices')

    characterVisuNode = characterMainNode.createChild('visu')
    characterVisuNode.createObject('OglModel', fileMesh=mesh_path + 'cube.obj', name='visual', mass='0.1')
    characterVisuNode.createObject('RigidMapping', template='Vec3d,ExtVec3f', input='@../model', output='@visual')

    #characterMainNode.createObject('BruteForceDetection', name='detection')
    #characterMainNode.createObject('NewProximityIntersection', name='proximity', alarmDistance='5.0', contactDistance='2.5')
    #characterMainNode.createObject('DefaultContactManager', name='response', response='default')
    #characterMainNode.createObject('EulerImplicitSolver', name='cg_odesolver', printLog='0')
    #characterMainNode.createObject('CGLinearSolver', template='graph_scattered', name='linear solver', iterations='40', tolerance='1e-009', threshold='1e-009')
    #characterMainNode.createObject('DefaultAnimationLoop', name='animation_loop')
    #characterMainNode.createObject('SceneColladaLoader', name='character', filename=mesh_path + 'character.dae', animationSpeed='0.14', generateCollisionModels='1')
    
    # clothes
    clothesMainNode = scene.createChild('clothes')
    
    clothesMainNode.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'poncho.obj')
    clothesMainNode.createObject('MechanicalObject', template='Vec3d', name='dof', position='@loader.position')
    clothesMainNode.createObject('UniformMass', template='Vec3d', mass='0.1')
    clothesMainNode.createObject('MeshTopology', name='mesh', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    clothesMainNode.createObject('BarycentricShapeFunction', template='ShapeFunction2d')
    clothesMainNode.createObject('TTriangleModel', template='Vec3d', name='models', proximity='0.2')
    
    clothesMainNode.createObject('MeshSpringForceField', linesStiffness='1.e+4', linesDamping='0.1')
    ##clothesMainNode.createObject('TriangularFEMForceField', template='Vec3d', mass='0.1') # replaced by the following components
    clothesDeformationNode = clothesMainNode.createChild('deformation')
    clothesDeformationNode.createObject('TopologyGaussPointSampler', name='sampler', inPosition='@../dof.position', showSamples='false', method='0', order='1')
    clothesDeformationNode.createObject('MechanicalObject', template='F321', name='triangleDeformationsDOF')
    clothesDeformationNode.createObject('LinearMapping', template="Mapping&lt;Vec3d,F321&gt;")
    
    #clothesStrainNode = clothesDeformationNode.createChild('strain')
    #clothesStrainNode.createObject('MechanicalObject', template='E321', name="StrainDOF")
    #clothesStrainNode.createObject('CorotationalStrainMapping', template='Mapping&lt;F321,E321&gt;', method="svd")
    #clothesStrainNode.createObject('HookeForceField', template='E321', youngModulus='2000', poissonRatio='0.2', viscosity='0.1')
	
    #clothesExtensionNode = clothesMainNode.createChild('extension')
    #clothesExtensionNode.createObject('MechanicalObject', template='Vec1d', name='ExtensionDOF')
    #clothesExtensionNode.createObject('EdgeSetTopologyContainer', template='Vec1d', name='ExtensionEdge', edges='@../mesh.edges')
    #clothesExtensionNode.createObject('ExtensionMapping', template='Vec3d,Vec1d', name="ExtensionMap", input='@../', output='@./')
    #clothesExtensionNode.createObject('UniformCompliance', template='Vec1d', name="UniComp", compliance='1e-4', isCompliance='0')
    
    clothesVisuNode = clothesMainNode.createChild('visu')
    clothesVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', mass='0.1')
    clothesVisuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f', mass='0.1', input='@../dof', output='@visual')