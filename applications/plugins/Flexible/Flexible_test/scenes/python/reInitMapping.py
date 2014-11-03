import Sofa
import SofaTest

flexible_path = Sofa.src_dir() + '/applications/plugins/Flexible'
mesh_path = flexible_path+ '/examples/beam/'

##Check if calling Mapping::init() change anything
#
#The trick is to know that if the option evaluateShapeFunction is activated
#in the ImageGaussPointSampler then a sampler::bwdInit() must be called
#to update weights using gauss points.
class Controller(SofaTest.Controller):
    def initGraph(self,node):
        self.success = 1
        self.count = 0
        return 0

    def createGraph(self,node):
        self.node = node
        return 0

    def initAndCheckMapping(self, node):
        mapping = node

        oldWeights = mapping.findData("weights").value
        oldWeightGradients = mapping.findData("weightGradients").value
        oldWeightHessians = mapping.findData("weightHessians").value

        mapping.init()

        newWeights = mapping.findData("weights").value
        newWeightGradients = mapping.findData("weightGradients").value
        newWeightHessians = mapping.findData("weightHessians").value

        if ( (oldWeights != newWeights) and (oldWeightGradients != newWeightGradients) and (oldWeightHessians != newWeightHessians) ):
            self.success = 0
        else:
            self.success = 1
        return 0

    def onEndAnimationStep(self,dt):
        return 0

    def onBeginAnimationStep(self,dt):
        self.count+=1
        if(self.count == 2):

            barycentricMapping = self.root.getChild("barycentricFrame").getChild("behavior").getObject("mapping")
            self.initAndCheckMapping(barycentricMapping)
            if(self.success == 0):
                self.sendFailure("(Barycentric Shape Function) calling init once again changed linearMapping weights for no reason")


            voronoiMapping = self.root.getChild("voronoiFrame").getChild("behavior").getObject("mapping")
            self.initAndCheckMapping(voronoiMapping)
            if(self.success == 0):
                self.sendFailure("(Voronoi Shape Function) calling init once again changed linearMapping weights for no reason")

            self.sendSuccess();
        return 0

def createBarycentricFrame( parentNode, name ):
    node = parentNode.createChild(name)
 
    #Solver
    node.createObject('EulerImplicit', name='integrator')
    node.createObject('CGLinearSolver', name='linearSolver', iteration='200', tolerance="1e-15", threshold='1.0e-15')

    #Frame
    dofPosition="0 1.0 -0.999 1 0 0 0 1 0 0 0 1 " + "0 1.0 0.999 1 0 0 0 1 0 0 0 1 "
    node.createObject('MechanicalObject', template='Affine', name='dofs', position=dofPosition, useMask="0", showObject='true,', showObjectScale='0.5')
    node.createObject('UniformMass', template='Affine',totalMass='0.01')
    #Constraint
    node.createObject('BoxROI', name='roi', template='Vec3d', box="-1 -2 -1.2 1 2 -0.8", drawBoxes='true', drawSize=1)
    node.createObject('FixedConstraint', indices="@[-1].indices")

    #Shape function
    node.createObject('MeshTopology', edges="0 0 0 1 1 1")
    node.createObject('BarycentricShapeFunction', name="shapeFunc")

    #Integration point sampling
    behaviorNode = node.createChild('behavior')
    behaviorNode.createObject("TopologyGaussPointSampler", name="sampler", inPosition="@../dofs.rest_position", showSamplesScale="0.1", drawMode="0")
    behaviorNode.createObject('MechanicalObject', name="intePts", template='F332', useMask="0", showObject="true", showObjectScale="0.05")
    behaviorNode.createObject('LinearMapping', name="mapping", template='Affine,F332', showDeformationGradientScale='0.2', showSampleScale="0", printLog="false")

    #Behavior
    eNode = behaviorNode.createChild('E')
    eNode.createObject( 'MechanicalObject', name='E', template='E332' )
    eNode.createObject( 'CorotationalStrainMapping', template='F332,E332', printLog='false' )
    eNode.createObject( 'HookeForceField', template='E332', youngModulus='100', poissonRatio='0', viscosity='0' )

	#Visu child node
    visuNode = node.createChild('Visu')
    visuNode.createObject('OglModel', template="ExtVec3f", name='Visual',filename=mesh_path+"beam.obj", translation="0 1 0")
    visuNode.createObject('LinearMapping', template='Affine,ExtVec3f')

def createVoronoiFrame( parentNode, name ):
    node = parentNode.createChild(name)
 
    #Solver
    node.createObject('EulerImplicit', name='integrator')
    node.createObject('CGLinearSolver', name='linearSolver', iteration='200', tolerance="1e-15", threshold='1.0e-15')

    #Frame
    node.createObject("MeshObjLoader", name="mesh", filename=mesh_path+"beam.obj", triangulate="1")
    node.createObject("ImageContainer", name="image", template="ImageUC", filename=mesh_path+"beam.raw", drawBB="false")
    node.createObject("ImageSampler", name="sampler", template="ImageUC", src="@image", method="1", param="0", fixedPosition="0 0 -0.999 0 0 0.999", printLog="false")
    node.createObject("MergeMeshes", name="merged", nbMeshes="2", position1="@sampler.fixedPosition", position2="@sampler.position")
    #node.createObject("ImageViewer", template="ImageB", name="viewer", src="@image")
    node.createObject('MechanicalObject', template='Affine', name='dofs', src="@merged", useMask="0", showObject='true,', showObjectScale='0.5')
    #Shape function
    node.createObject('VoronoiShapeFunction', name="shapeFunc", position='@dofs.rest_position', src='@image', useDijkstra="true", method="0", nbRef="4")
    #Uniform Mass
    node.createObject('UniformMass', template='Affine',totalMass='0.01')
    #Constraint
    node.createObject('BoxROI', name='roi', template='Vec3d', box="-1 -2.0 -1.2 1 2.0 -0.8", drawBoxes='true', drawSize=1)
    node.createObject('FixedConstraint', indices="@[-1].indices")

    #Gauss point sampling
    behaviorNode = node.createChild('behavior')
    behaviorNode.createObject('ImageGaussPointSampler', name='sampler', indices='@../shapeFunc.indices', weights='@../shapeFunc.weights', transform='@../shapeFunc.transform', method='2', order='4', targetNumber='1', printLog='false', showSamplesScale=0.1, drawMode=0, evaluateShapeFunction="false")
    behaviorNode.createObject('MechanicalObject', name="intePts", template='F332', useMask="0", showObject="false", showObjectScale="0.05")
    behaviorNode.createObject('LinearMapping', name="mapping", template='Affine,F332', assembleJ='true', showDeformationGradientScale='0.2', printLog="false")

    #Behavior
    eNode = behaviorNode.createChild('E')
    eNode.createObject( 'MechanicalObject', name='E', template='E332' )
    eNode.createObject( 'CorotationalStrainMapping', template='F332,E332', printLog='false' )
    eNode.createObject( 'HookeForceField', template='E332', youngModulus='100', poissonRatio='0', viscosity='0' )

	#Visu child node
    visuNode = node.createChild('Visu')
    visuNode.createObject('OglModel', template="ExtVec3f", name='Visual',filename=mesh_path+"beam.obj")
    visuNode.createObject('LinearMapping', template='Affine,ExtVec3f')

    return node

def createScene( root ) :
    #Root node data
    root.findData('dt').value=0.001
    root.findData('gravity').value='0 -10 0'

    #Required setting
    root.createObject('RequiredPlugin', name="flexible", pluginName='Flexible', printLog="false")
    root.createObject('RequiredPlugin', name="image", pluginName='image', printLog="false")

    #VisuStyle
    root.createObject('VisualStyle', name='visuStyle', displayFlags='showWireframe showBehaviorModels')

    #Animation Loop
    root.createObject('DefaultAnimationLoop');
    root.createObject('DefaultVisualManagerLoop');

    #Python Script Controller
    root.createObject('PythonScriptController', filename = __file__, classname='Controller')

    createVoronoiFrame(root, 'voronoiFrame');
    createBarycentricFrame(root, 'barycentricFrame');
