import sys,os
import Sofa
import Flexible.IO
from Compliant import Tools as Tools

# main path
currentdir = os.path.dirname(os.path.realpath(__file__))+'/'
__file = __file__.replace('\\', '/')  # windows compatible filename
source = './mesh/source_cage.obj' 
target = './mesh/target_cage.obj' 
source_skeleton = './mesh/source_skeleton.obj'
target_skeleton = './mesh/target_skeleton.obj'
voxel_size = 0.0075

# ================================================================= #
# Method called in Sofa
# ================================================================= #
def createScene(root_node) :     
    # Required plugin
    root_node.createObject('RequiredPlugin', pluginName='image')
    root_node.createObject('RequiredPlugin', pluginName='Flexible')
    root_node.createObject('RequiredPlugin', pluginName='Compliant')
    root_node.createObject('RequiredPlugin', pluginName='RigidScale')

    # Script launch
    root_node.createObject('PythonScriptController', name='script', filename=__file, classname='MyClass')

# ================================================================= #
# Creation of the scene
# ================================================================= #
class MyClass(Sofa.PythonScriptController):
         
    def createGraph(self, root):
        # Variable
        self.E_t = 0
        self.E_t_dt = 0
        self.root_node = root
        
        # Sofa parameters
        self.root_node.createObject('BackgroundSetting',color='1 1 1')
        self.root_node.createObject('VisualStyle', displayFlags='showVisual hideWireframe showBehaviorModels hideForceFields hideInteractionForceFields')
        self.root_node.createObject('StaticSolver')
        self.root_node.createObject('CGLinearSolver', iterations=500, tolerance=1E-10, threshold=1E-10)
        self.root_node.findData('gravity').value = '0 0 0'

        # Object to transfer creation
        node = self.root_node.createChild('main')
        node.createObject('MeshObjLoader',name='source', filename=source, triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='0.01 0.01 0.01')
        node.createObject('MeshObjLoader',name='source_skeleton', filename=source_skeleton, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        node.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', insideValue='1', voxelSize=voxel_size, padSize=2, rotateImage='false')
        node.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false')
        node.createObject('ImageSampler', template='ImageUC', name='sampler', src='@image', method=1, param='15 0', clearData=0)
        node.createObject('MeshTopology', name='frame_topo', position='@sampler.position') 
                
        #================================ Target model ===================================
        targetNode = node.createChild('target') 
        targetNode.createObject('MeshObjLoader', name='target', filename=target_skeleton, translation='2 0 0', rotation='0 0 0', scale3d='1 1 1', showObject=0)        
        targetNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', src='@target', showObject=0)
        targetNode.createObject('FixedConstraint', fixAll='1' )          
        targetVisuNode = targetNode.createChild('visu')
        targetVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../target', color='0.5 0.5 0.5 0.75') 
        
        #================================ Rigid frame ====================================
        rigidNode = node.createChild('rigid')
        rigidNode.createObject('MechanicalObject', template='Rigid3d', name='DOFs', src='@../frame_topo', showObject=1, showObjectScale='0.025')
        rigidNode.createObject('PartialFixedConstraint', name="partialFixedConstraint", fixAll='1', fixedDirections="0 0 0 1 1 1")
                
        #=================================== Scale =======================================
        scaleNode = node.createChild('scale')
        scaleNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', showObject=0, showObjectScale='0.1')

        #============================= Registration model ================================
        objMainNode = rigidNode.createChild('deformable')
        scaleNode.addChild(objMainNode)

        # scene creation
        objMainNode.createObject('MeshObjLoader',name='source', filename=source, triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='0.01 0.01 0.01')
        objMainNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=voxel_size, padSize=0, rotateImage='false')
        objMainNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false') 
        objMainNode.createObject('MechanicalObject', template='Affine', name='parent', src='@../../frame_topo', showObject=0, showObjectScale='0.1')
        objMainNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1='@../../rigid/DOFs', input2='@../../scale/DOFs', output='@.', autoInit='1', printLog='0')
        objMainNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@parent.rest_position', image='@image.image', transform='@image.transform', nbRef=8, clearData=1, bias=0) 
        # Behavior
        objBehaviorNode = objMainNode.createChild('behavior')
        objBehaviorNode.createObject('ImageGaussPointSampler', name='sampler', indices='@../SF.indices', weights='@../SF.weights', transform='@../SF.transform', method=2, order=1, targetNumber=1000)
        objBehaviorNode.createObject('MechanicalObject', template='F331')
        objBehaviorNode.createObject('LinearMapping', template='Affine,F331')      
        objBehaviorNode.createObject('ProjectiveForceField', template='F331', youngModulus=1E3, poissonRatio=0, viscosity=0) 
        # Contact        
        objContactNode = objMainNode.createChild('registration')     
        objContactNode.createObject('MeshTopology', name='topo', src='@../source')
        objContactNode.createObject('MechanicalObject', name='DOFs')
        objContactNode.createObject('UniformMass', totalMass=1)
        objContactNode.createObject('TriangleModel')
        objContactNode.createObject('LinearMapping', template='Affine,Vec3d')
        # Visual model
        objVisuNode = objContactNode.createChild('visual')
        objVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../topo', color='1 0.2 0.2 0.9')
        objVisuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f') 
        objVisuOriginNode = objContactNode.createChild('origin_visual')
        objVisuOriginNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../topo', color='0.2 0.2 0.8 0.9')
        # Registration
        objRegistrationNode = objMainNode.createChild('force')
        loader = objRegistrationNode.createObject('MeshObjLoader',name='source_skeleton', filename=source_skeleton, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        objRegistrationNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', src='@source_skeleton')
        objRegistrationNode.createObject('LinearMapping', template='Affine,Vec3d')
        # registration force field
        springs = ""
        for i in range(len(loader.position)):
            springs += str(i)+' '+str(i)+' '

        distanceNode = objRegistrationNode.createChild('registration_constraint')
        targetNode.addChild(distanceNode)
        distanceNode.createObject('MechanicalObject', template='Vec3d', name='distanceDOFs')
        distanceNode.createObject('DifferenceMultiMapping', template='Vec3d,Vec3d', input='@'+Tools.node_path_rel(distanceNode, targetNode)+' @'+Tools.node_path_rel(distanceNode, objRegistrationNode), output='@.', pairs=springs, showObjectScale="0.005")
        distanceNode.createObject('UniformCompliance', name='constraint', isCompliance=0, compliance=1E-6, damping=0.1)
