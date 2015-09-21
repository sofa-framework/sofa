0000000000.
import sys,os
import Sofa
import Flexible.IO
import math
import time

# main path
__file = __file__.replace('\\', '/')  # windows compatible filename
currentdir = os.path.dirname(os.path.realpath(__file__))+'/'
currentdir = currentdir.replace('\\', '/')
datadir = currentdir + '/data/'

def createScene(root_node) :
    # Required plugin
    root_node.createObject('RequiredPlugin', pluginName='SofaPython')
    root_node.createObject('RequiredPlugin', pluginName='Flexible')
    root_node.createObject('RequiredPlugin', pluginName='Compliant')
    root_node.createObject('RequiredPlugin', pluginName='ModalSubspace')
    root_node.createObject('PythonScriptController', name='script', filename=__file, classname='MyClass')

class MyClass(Sofa.PythonScriptController):

    def createGraph(self, root):
        self.percent = 0.2
        self.parentNode = root
        self.n = 0
        self.r = int( self.percent * 675 )
        reducedDDL=self.r
        print "nombre de training positions a fournir: ", reducedDDL
        self.t = 0
        self.index = 0
        self.table = [[0.0]]*self.r
        self.amplitude = [[0.0]]*self.r
        self.stepClicked = 0
        self.beginTime = 0
        self.endTime = 0
        self.deltaTime = 0

        ###############################################################################
        self.parentNode.findData('gravity').value = '0.0, 0.0, 0.0'
        self.parentNode.createObject('VisualStyle',  displayFlags='showBehavior hideVisual')

        # On doit charger le plugin ModalSubspace car ensuite on utilise...
        # ...le SubspaceMapping.inl
        # Pour animer la scene
        self.parentNode.createObject('CompliantImplicitSolver')
        self.parentNode.createObject('LDLTSolver' , printLog=0)
        self.parentNode.createObject('LDLTResponse' , printLog=0)
        #
        #
        main = self.parentNode.createChild('Main')
        #
        # position = "autant de zeros que de modes propres choisis"
        base = '0'+' '
        posDDL = base*(self.r-1)+'0'
        main.createObject('MechanicalObject' ,  name='master' , template='Vec1d' , position=posDDL)
        #
        #
        child = main.createChild('Child')
        # Subdivided cube
        regularGridTopology = child.createObject('RegularGridTopology' , name='grid' , n='10 5 5' , min='0 0 0' , max='5 1 1')
        n = regularGridTopology.n[0]

        child.createObject('MechanicalObject', name='slave' , template='Vec3d' , showObject='true', showObjectScale="5",  showIndices="0",  showIndicesScale="0.0003")
        child.createObject('UniformMass' , name='themass' , totalMass='1')

        # polynomes de Lagrange
        child.createObject('BarycentricShapeFunction', template="ShapeFunctiond")

        child.createObject('SubspaceMapping' , template='Vec1d,Vec3d' , input='@../master' , output='@slave' , filename ='../../../src/sofa-dev/applications-dev/plugins/ModalSubspace/matrices/maMatriceU.txt')


        behavior = child.createChild('Behavior')
        behavior.createObject('TopologyGaussPointSampler', name="sampler", inPosition="@../grid.position", showSamplesScale="0", method="0", order="2")
        mechanicalObject = behavior.createObject('MechanicalObject',  template="F331", name="F", showObject="0", showObjectScale="0.05")
        behavior.createObject('LinearMapping', name="linMap", template="Vec3d,F331", showDeformationGradientScale="0.05",  showDeformationGradientStyle="0 - All axis",  showColorOnTopology="2 - sqrt(det(F^T.F))-1")


        strain = behavior.createChild('Strain')
        mechanicalObject = strain.createObject('MechanicalObject', name="IPObject", template="E331")
        strain.createObject('CorotationalStrainMapping', name="Corot", template="F331,E331")
        cubatureHookeForceField = strain.createObject('CubatureHookeForceField',  template="E331", youngModulus="10000", poissonRatio="0.3", switchMode="false", cubature="false", optimizedCubature="true", integrationPointsPercent="0.2", nbTrainingPositions="10")
        cubatureHookeForceField.nbElements = (n[0]-1)*(n[1]-1)*(n[2]-1)


        cubatureHookeForceField.x1File = currentdir + "/../../../Cubature_test/x1_20.dat"
        cubatureHookeForceField.v1File = currentdir + "/../../../Cubature_test/v1_20.dat"

        cubatureHookeForceField.potentialEnergyFile = currentdir + "/../../../Cubature_test/potentialEnergy20.dat"

        #cubature = strain.createObject('Cubature', name="cub", template="E331", stress="@IPObject.force", listening="true", naiveCubature="true", integrationPointsPercent="0.05")
        #cubature.nbElements = (n[0]-1)*(n[1]-1)*(n[2]-1)
        #print "cubature.r = ", cubature.r

        return self.parentNode

    # Position de depart fixee
    def initGraph(self,root):
        self.node = root
        self.MechanicalState = root.getObject('Main/master')
        return 0

    def setAmplitudeToOscilloAtIndex(self, i, v):
        self.amplitude[int(i)] = [v]
        sys.stdout.flush()

    def onBeginAnimationStep(self, dt):
        if self.t==0.0:
            for i in range(0,self.r):
                self.table[i] = [self.amplitude[i][0]]#*math.sin(3*self.t)]

            self.MechanicalState.position = self.table

        self.t = self.t + dt
        print self.t
