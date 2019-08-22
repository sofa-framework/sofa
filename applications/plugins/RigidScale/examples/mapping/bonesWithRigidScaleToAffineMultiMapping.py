import sys,os
import Sofa
import numpy
import Flexible.IO
from Compliant import Tools as Tools
import SofaPython
import SofaPython.Quaternion
from SofaPython.Tools import listToStr as listToStr, listListToStr as listListToStr

# main path
currentdir = os.path.dirname(os.path.realpath(__file__))+'/'
__file = __file__.replace('\\', '/')  # windows compatible filename
p1 = [2.143849, 10.597820, 0.624865]
p2 = [1.667256, 13.205175, 0.711915]
j1 = ((numpy.array(p1) + numpy.array(p2))/2).tolist()
dofs_position = [p1, p2]
joint_position = [j1]
joint_orientation = SofaPython.Quaternion.from_line((numpy.array(p1)-numpy.array(p2)).tolist(), -1, 2)
offset_position = [(numpy.array(j1) - numpy.array(p1)).tolist(), (numpy.array(j1) - numpy.array(p2)).tolist()]

# ================================================================= #
# Method called in Sofa
# ================================================================= #
def createScene(root_node) :     
    # Required plugin
    root_node.createObject('RequiredPlugin', pluginName='image')
    root_node.createObject('RequiredPlugin', pluginName='Flexible')
    root_node.createObject('RequiredPlugin', pluginName='Compliant')
    root_node.createObject('RequiredPlugin', pluginName='SohusimDev')
    root_node.createObject('RequiredPlugin', pluginName='Registration')
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
        self.root_node.findData('gravity').value = '0 0 0'
        
        # Sofa parameters
        self.root_node.createObject('BackgroundSetting',color='1 1 1')
        self.root_node.createObject('VisualStyle', displayFlags='showVisual hideWireframe showBehaviorModels hideForceFields showInteractionForceFields')

        self.root_node.createObject('EulerImplicit',rayleighStiffness='0.01',rayleighMass='0.01')
        self.root_node.createObject('CGLinearSolver', iterations=200, tolerance=1E-6, threshold=1E-6)
        
        # Object to transfer creation
        node = self.root_node.createChild('main')
        node.createObject('MeshObjLoader',name='source', filename='./mesh/source.obj', triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        node.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', insideValue='1', voxelSize=0.01, padSize=2, rotateImage='false')
        node.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false')
        node.createObject('MeshTopology', name='frame_topo', position=listListToStr(dofs_position))

        #================================ Target model ===================================
        targetNode = node.createChild('target')
        targetNode.createObject('MeshObjLoader', name='loader', filename='./mesh/target.obj', rotation='0 0 0', translation='0 0 0', scale3d='1 1 1', showObject=0)
        targetNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', src='@loader', showObject=0)
        targetNode.createObject('NormalsFromPoints', name='normalsFromPoints', template='Vec3d', position='@DOFs.position', triangles='@loader.triangles', invertNormals=0)
        targetNode.createObject('FixedConstraint', fixAll='1', drawSize=0.001)
        targetVisuNode = targetNode.createChild('visu')
        targetVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../loader', color='0.5 0.5 0.5 0.85')

        #================================ Rigid frame ====================================
        rigidNode = node.createChild('rigid')           
        rigidNode.createObject('MechanicalObject', template='Rigid', name='DOFs', src='@../frame_topo', showObject=1, showObjectScale='0.1')
        rigidNode.createObject('RigidMass', mass="1 1 ", inertia="1 1 1  1 1 1")
        #rigidNode.createObject('FixedConstraint', name='fixed', indices='0')

        #=================================== Scale =======================================
        scaleNode = node.createChild('scale')
        scaleNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', position='1 1 1 1 1 1', showObject=0, showObjectScale='0.1')

        #=========================== Alignement constraint ===============================
        offsetNode = rigidNode.createChild('offset')
        offsetNode.createObject('MechanicalObject', template='Rigid', name='DOFs', position='0 0 0 0 0 0 1  0 0 0 0 0 0 1', showObject=1, showObjectScale='0.1')
        offsetNode.createObject('AssembledRigidRigidMapping', template='Rigid,Rigid', source='0 '+listToStr(offset_position[0]) + listToStr(joint_orientation) + ' 1 '+listToStr(offset_position[1]) + listToStr(joint_orientation))
        # --- old things even if they don't work well are often more stable, just for creating the prototype ...
        offsetNode.createObject('JointSpringForceField', template='Rigid', name='joint', object1='@.', object2='@.', spring=' BEGIN_SPRING 0 1 FREE_AXIS 0 1 0 0 0 0 KS_T 0 1E10 KS_R 0 1e10 KS_B 3E3 KD 0.1 R_LIM_X 0 0 R_LIM_Y 0 0 R_LIM_Z 0 0 REST_T 0 1 0 END_SPRING')

        #============================= Registration model ================================
        objMainNode = rigidNode.createChild('deformable')
        scaleNode.addChild(objMainNode)
        # scene creation
        loader = objMainNode.createObject('MeshObjLoader',name='source', filename='./mesh/source.obj', triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        objMainNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=0.01, padSize=0, rotateImage='false')
        objMainNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false') 
        objMainNode.createObject('MechanicalObject', template='Affine', name='parent', showObject=1, src='@../../frame_topo', showObjectScale='0.1')
        objMainNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1='@../../rigid/DOFs', input2='@../../scale/DOFs', output='@.', autoInit='1', printLog='0')
        objMainNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@parent.rest_position', image='@image.image', transform='@image.transform', nbRef=4, clearData=1, bias=0)
        # Behavior Node
        objBehaviorNode = objMainNode.createChild('behavior')
        objBehaviorNode.createObject('ImageGaussPointSampler', name='sampler', indices='@../SF.indices', weights='@../SF.weights', transform='@../SF.transform', method=2, order=1, targetNumber=40)
        objBehaviorNode.createObject('MechanicalObject', template='F331')  
        objBehaviorNode.createObject('LinearMapping', template='Affine,F331')    
        objBehaviorNode.createObject('ProjectiveForceField', template='F331', youngModulus=1E6, poissonRatio=0, viscosity=0, isCompliance=0)
        # Contact
        objContactNode = objMainNode.createChild('registration')     
        objContactNode.createObject('MeshTopology', name='topo', src='@../source')
        objContactNode.createObject('MechanicalObject', name='DOFs')
        objContactNode.createObject('UniformMass', totalMass=1)
        objContactNode.createObject('TriangleModel')
        objContactNode.createObject('LinearMapping', template='Affine,Vec3d')
        # Visual model
        objVisuNode = objContactNode.createChild('visual')
        objVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../topo', color='1 0.2 0.2 0.8')
        objVisuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f')
        # Registration : first attraction field to well positioning the source on the target
        objRegistrationNode = objMainNode.createChild('force')
        objRegistrationNode.createObject('MeshTopology', name='topo', src='@../source')
        objRegistrationNode.createObject('MechanicalObject', template='Vec3d', name='DOFs')
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
        # Registration : surface to surface registration forces
        surfaceRegistrationNode = objMainNode.createChild('reg_force')
        surfaceRegistrationNode.createObject('MeshTopology', name='topo', src='@../source')
        surfaceRegistrationNode.createObject('MechanicalObject', name='DOFs')
        surfaceRegistrationNode.createObject('Triangle')
        surfaceRegistrationNode.createObject('LinearMapping', template='Affine,Vec3d', assemble=0)
        surfaceRegistrationNode.createObject('NormalsFromPoints', name='normalsFromPoints', template='Vec3d', position='@DOFs.position', triangles='@topo.triangles', invertNormals=0)
        surfaceRegistrationNode.createObject('ClosestPointRegistrationForceField', name='ICP', template='Vec3d'
                                                                                             , sourceTriangles='@topo.triangles', sourceNormals='@normalsFromPoints.normals'
                                                                                             , position='@../../../target/loader.position' , triangles='@../../../target/loader.triangles', normals='@../../../target/normalsFromPoints.normals'
                                                                                             , cacheSize=4, blendingFactor=1, stiffness=1E3, damping=1E-3
                                                                                             , outlierThreshold=0, normalThreshold=0, rejectOutsideBbox=0, drawColorMap='0')



