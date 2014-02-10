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
    
    scene.createObject('AssembledSolver', stabilization='1')
    #scene.createObject('MinresSolver', iterations='500')
    scene.createObject('SequentialSolver', precision='1e-08', relative='false', iterations='50')
    scene.createObject('DefaultPipeline', depth='6')
    scene.createObject('BruteForceDetection')
    scene.createObject('DefaultContactManager', responseParams='damping=0&amp;compliance=0&amp;restitution=0.5', response='CompliantContact')
    
    # character (currently we use a fixed box)
    characterMainNode = scene.createChild('character')
    
    characterMainNode.createObject('MechanicalObject', template='Rigid', name='model', position='0 -3.0 0 0 0 0 1')
    characterMainNode.createObject('RigidMass', mass='1', inertia='1 1 1')
    characterMainNode.createObject('FixedConstraint', indices='0')
    
    characterCollisionNode = characterMainNode.createChild('collision')
    characterCollisionNode.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'cube.obj')
    characterCollisionNode.createObject('MeshTopology', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    characterCollisionNode.createObject('MechanicalObject', template='Vec3d', name='vertices', position='@loader.position')
    characterCollisionNode.createObject('TTriangleModel', template='Vec3d',  simulated='0', moving='0')
    characterCollisionNode.createObject('RigidMapping', template='Rigid,Vec3d', input='@../model', output='@./vertices')

    characterVisuNode = characterMainNode.createChild('visu')
    characterVisuNode.createObject('OglModel', fileMesh=mesh_path + 'cube.obj', name='visual', mass='0.1')
    characterVisuNode.createObject('RigidMapping', template='Vec3d,ExtVec3f', input='@../model', output='@visual')
    
#    <Node name="ground">
#        <MechanicalObject name="dofs" template="Rigid" position="0 -3.2 0 0 0 0 1" />
#        <RigidMass mass="1" inertia="1 1 1" />
#        <Node name="visual" tags="Visual">
#            <OglModel fileMesh="mesh/cube.obj" name="model" scale3d="20 2 20" material="Default Diffuse 1 0.5 0.5 0.5 1 Ambient 1 0.1 0.1 0.1 1 Specular 0 0.5 0.5 0.5 1 Emissive 0 0.5 0.5 0.5 1 Shininess 0 45 " primitiveType="DEFAULT" />
#            <RigidMapping input="@.." template="Rigid,ExtVec3f" output="@model" />
#        </Node>
#        <Node name="collision">
#            <MeshObjLoader filename="mesh/cube.obj" name="loader" scale3d="20 2 20" />
#            <MeshTopology tetrahedra="@loader.tetras" name="topology" position="@loader.position" hexahedra="@loader.hexas" edges="@loader.edges" quads="@loader.quads" triangles="@loader.triangles" />
#            <MechanicalObject name="vertices" template="Vec3d" position="@loader.position" />
#            <TTriangleModel template="Vec3d" simulated="0" moving="0" />
#            <RigidMapping input="@../" template="Rigid,Vec3d" output="@./" />
#        </Node>
#        <FixedConstraint indices="0" />
#    </Node>

    #characterNode.createObject('BruteForceDetection', name='detection')
    #characterNode.createObject('NewProximityIntersection', name='proximity', alarmDistance='5.0', contactDistance='2.5')
    #characterNode.createObject('DefaultContactManager', name='response', response='default')
    #characterNode.createObject('EulerImplicitSolver', name='cg_odesolver', printLog='0')
    #characterNode.createObject('CGLinearSolver', template='graph_scattered', name='linear solver', iterations='40', tolerance='1e-009', threshold='1e-009')
    #characterNode.createObject('DefaultAnimationLoop', name='animation_loop')
    #characterNode.createObject('SceneColladaLoader', name='character', filename=mesh_path + 'character.dae', animationSpeed='0.14', generateCollisionModels='1')
    

    # clothes
    clothesMainNode = scene.createChild('clothes')
    
    clothesMainNode.createObject('MeshObjLoader', name='loader', filename=mesh_path + 'poncho.obj')
    clothesMainNode.createObject('MechanicalObject', template='Vec3d', name='dof', position='@loader.position')
    clothesMainNode.createObject('UniformMass', template='Vec3d', mass='0.1')
    clothesMainNode.createObject('MeshTopology', position='@loader.position', edges='@loader.edges', triangles='@loader.triangles', quads='@loader.quads', tetrahedra='@loader.tetras', hexahedra='@loader.hexas')
    clothesMainNode.createObject('BarycentricShapeFunction', template='ShapeFunction2d')
    clothesMainNode.createObject('TTriangleModel', template='Vec3d', name='models')
    
    #clothesMainNode.createObject('TriangularBendingSprings', template='Vec3d', name='springs', stiffness='50', damping='1')
    
    #clothesMainNode.createObject('TriangularFEMForceField', template='Vec3d', mass='0.1') # replaced by the following components
    clothesDeformationNode = clothesMainNode.createChild('deformation')
    clothesDeformationNode.createObject('TopologyGaussPointSampler', name='sampler', inPosition='@../dof.position', showSamples='false', method='0', order='1')
    clothesDeformationNode.createObject('MechanicalObject', template='F321', name='triangleDeformationsDOF')
    clothesDeformationNode.createObject('LinearMapping', template="Mapping&lt;Vec3d,F321&gt;")
    
    clothesStrainNode = clothesDeformationNode.createChild('strain')
    clothesStrainNode.createObject('MechanicalObject', template='E321', name="StrainDOF")
    clothesStrainNode.createObject('CorotationalStrainMapping', template='Mapping&lt;F321,E321&gt;', method="svd")
    clothesStrainNode.createObject('HookeForceField', template='E321', youngModulus='2000', poissonRatio='0.2', viscosity='0.1')
    
    clothesVisuNode = clothesMainNode.createChild('visu')
    clothesVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', mass='0.1')
    clothesVisuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f', mass='0.1', input='@../dof', output='@visual')