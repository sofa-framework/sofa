import sys,os
import Sofa
import Flexible.IO

# main path
currentdir = os.path.dirname(os.path.realpath(__file__))+'/'
__file = __file__.replace('\\', '/')  # windows compatible filename 
source = './mesh/monkey.obj'
voxel_size = 0.05

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
        self.t = 0
        self.root_node = root
        
        # Sofa parameters
        self.root_node.createObject('BackgroundSetting',color='1 1 1')
        self.root_node.createObject('VisualStyle', displayFlags='showVisual hideWireframe hideBehaviorModels hideForceFields hideInteractionForceFields')

        # Constraint components
        root.createObject('CompliantAttachButtonSetting' )
        root.createObject('CompliantImplicitSolver', stabilization=1)
        root.createObject('SequentialSolver', iterations=500, precision=1E-15, iterateOnBilaterals=1)
        root.createObject('LDLTResponse', schur=0)

        self.root_node.findData('gravity').value = '0 0 0'

        # Object to transfer creation
        node = self.root_node.createChild('monkey')
        node.createObject('MeshObjLoader',name='source', filename=source, triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        node.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', insideValue='1', voxelSize=voxel_size, padSize=2, rotateImage='false')
        node.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false')
        node.createObject('ImageSampler', template='ImageUC', name='sampler', src='@image', method=1, param='4 0', clearData=0)
        node.createObject('MeshTopology', name='imposed', position='0 0 0')
        node.createObject('MergeMeshes', name='frame_topo', nbMeshes=2, position1='@imposed.position',  position2='@sampler.position')

                        
        #================================ Rigid frame ====================================
        rigidNode = node.createChild('rigid')
        rigidNode.createObject('MechanicalObject', template='Rigid3d', name='DOFs', src='@../frame_topo', showObject=0, showObjectScale='0.1')
        rigidNode.createObject('FixedConstraint', indices='1' )
               
        #=================================== Scale =======================================
        scaleNode = node.createChild('scale')
        self.scaleDOF = scaleNode.createObject('MechanicalObject', template='Vec3d', name='DOFs', position='1 1 1', showObject=0, showObjectScale='0.1')
        positiveNode = scaleNode.createChild('positive')
        positiveNode.createObject('MechanicalObject', template='Vec3d', name='positivescaleDOFs')
        positiveNode.createObject('DifferenceFromTargetMapping', template='Vec3d,Vec3d', applyRestPosition=1, targets="1E-3 1E-3 1E-3")
        positiveNode.createObject('UniformCompliance', isCompliance=1, compliance=0)
        positiveNode.createObject('UnilateralConstraint')
        positiveNode.createObject('Stabilization', name='Stabilization')

        #================== offsets mapped to both rigid and scale =======================
        offsetNode = rigidNode.createChild('offset')
        scaleNode.addChild(offsetNode)
        offsetNode.createObject('MechanicalObject', template='Rigid3d', name='DOFs', position='0 -1.5 0 0 0 0 1', showObject=1, showObjectScale='0.25')
        offsetNode.createObject('RigidScaleToRigidMultiMapping', template='Rigid,Vec3d,Rigid', input1='@../../rigid/DOFs', input2='@../../scale/DOFs', output='@.', index='0 0 0', printLog='0')

        #============================= Registration model ================================
        self.affineNode = rigidNode.createChild('affine')
        scaleNode.addChild(self.affineNode)

        # Scene creation
        self.affineNode.createObject('MeshObjLoader',name='source', filename=source, triangulate=1, translation='0 0 0', rotation='0 0 0', scale3d='1 1 1')
        self.affineNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@source', value=1, insideValue=1, voxelSize=voxel_size, padSize=0, rotateImage='false')
        self.affineNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false')
        self.affineNode.createObject('MechanicalObject', template='Affine', name='parent', showObject=1, showObjectScale='0.1')
        # Automatic initialization of the mapping and component related to the mapping
        self.affineNode.createObject('RigidScaleToAffineMultiMapping', template='Rigid,Vec3d,Affine', input1='@../../rigid/DOFs', input2='@../../scale/DOFs', output='@.', autoInit='1', printLog='0')
        self.affineNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@parent.rest_position', image='@image.image', transform='@image.transform', nbRef=4, clearData=1, bias=0)
        # Behavior Node
        objBehaviorNode = self.affineNode.createChild('behavior')
        objBehaviorNode.createObject('ImageGaussPointSampler', name='sampler', indices='@../SF.indices', weights='@../SF.weights', transform='@../SF.transform', method=2, order=1, targetNumber=200)
        objBehaviorNode.createObject('MechanicalObject', template='F331')
        objBehaviorNode.createObject('LinearMapping', template='Affine,F331')
        objBehaviorNode.createObject('ProjectiveForceField', template='F331', youngModulus=1E4, poissonRatio=0.1, viscosity=0.1)
        # Mass
        self.affineMassNode = self.affineNode.createChild('mass')
        self.affineMassNode.createObject('TransferFunction', name='density', template='ImageUC,ImageD', inputImage='@../image.image', param='0 0 1 '+str(2000))
        self.affineMassNode.createObject('MechanicalObject', template='Vec3d')
        self.affineMassNode.createObject('LinearMapping', template='Affine,Vec3d')
        self.affineMassNode.createObject('MassFromDensity',  name='MassFromDensity', template='Affine,ImageD', image='@density.outputImage', transform='@../image.transform', lumping='0')
        self.affineNode.createObject('AffineMass', massMatrix='@mass/MassFromDensity.massMatrix')
        # Contact
        objContactNode = self.affineNode.createChild('registration')
        objContactNode.createObject('MeshTopology', name='topo', src='@../source')
        objContactNode.createObject('MechanicalObject', name='DOFs')
        objContactNode.createObject('TriangleModel')
        objContactNode.createObject('LinearMapping', template='Affine,Vec3d')
        # Visual model
        objVisuNode = objContactNode.createChild('visual')
        objVisuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../topo', color='1 0.2 0.2 1')
        objVisuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f')
        return node

    # To remove warning
    def onBeginAnimationStep(self, dt):
        if self.t==0 :
            self.affineMassNode.active = False
        self.t = self.t+dt

    def onEndAnimationStep(self, dt):
        isNegative = False
        for p in self.scaleDOF.position:
            for xi in  p:
                if xi <= 0 :
                    isNegative = True
                    print "value : ", xi
        if isNegative :
            print "Negative values still exist : ", self.t