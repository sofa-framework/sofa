import sys,os
import Sofa
import Flexible.IO
from Compliant import Tools as Tools

# main path
currentdir = os.path.dirname(os.path.realpath(__file__))+'/'
__file = __file__.replace('\\', '/')  # windows compatible filename 
source = './mesh/cube.obj'
target = './mesh/cube.obj'
voxel_size = 0.1

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
        node = self.root_node.createChild('cube_source')
        node.createObject('MeshObjLoader',name='source', filename=source, triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        node.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', insideValue='1', voxelSize=voxel_size, padSize=0, rotateImage='false')
        node.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false')
        node.createObject('ImageSampler', template='ImageUC', name='sampler', src='@image', method=1, param='1 0', clearData=0)
        node.createObject('MeshTopology', name='frame_topo', position='@sampler.position') 
                
        #================================ Target model ===================================
        targetNode = node.createChild('target') 
        targetNode.createObject('MeshObjLoader', name='target', filename=target, triangulate=1, translation='2 0 0', rotation='0 90 45', scale3d='1.5 3 2', showObject=0)
        targetNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', src='@target', showObject=0)
        targetNode.createObject('FixedConstraint', fixAll='1' )          
        targetVisuNode = targetNode.createChild('visu')
        targetVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../target', color='0.5 0.5 0.5 0.75') 
        
        #=================================== Scale =======================================
        scaleNode = node.createChild('scale')
        scaleNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', position='1 1 1', showObject=0, showObjectScale='0.1')
        
        #================================ Rigid frame ====================================
        rigidNode = node.createChild('rigid')
        rigidNode.createObject('MechanicalObject', template='Rigid3d', name='DOFs', src='@../frame_topo', showObject=0, showObjectScale='0.1')

        #================== offsets mapped to both rigid and scale =======================
        offsetNode = rigidNode.createChild('offset')
        scaleNode.addChild(offsetNode)
        offsetNode.createObject('MechanicalObject', template='Rigid3d', name='DOFs', position='0 1 0 0 0 0 1', showObject=1, showObjectScale='0.25')
        offsetNode.createObject('RigidScaleToRigidMultiMapping', template='Rigid,Vec3d,Rigid', input1='@../../rigid/DOFs', input2='@../../scale/DOFs', output='@.', index='0 0 0', printLog='0')
        
        #============================= Registration model ================================
        objMainNode = rigidNode.createChild('main')
        scaleNode.addChild(objMainNode)

        # scene creation
        loader = objMainNode.createObject('MeshObjLoader',name='source', filename=source, triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        objMainNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=voxel_size, padSize=0, rotateImage='false')
        objMainNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false') 
        objMainNode.createObject('MechanicalObject', template='Affine', name='parent', src='@../../frame_topo', showObject=1, showObjectScale='0.1')
        objMainNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1='@../../rigid/DOFs', input2='@../../scale/DOFs', output='@.', index='0 0 0', printLog='0')
        objMainNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@parent.rest_position', image='@image.image', transform='@image.transform', nbRef=4, clearData=1, bias=0)  
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
        # Registration
        objRegistrationNode = objMainNode.createChild('force')
        objRegistrationNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', src='@../source')
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