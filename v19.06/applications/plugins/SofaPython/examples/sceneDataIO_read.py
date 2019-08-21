import sys, os, platform, math
import Sofa

import Flexible.IO
import Flexible.sml

import SofaPython.Tools
from SofaPython.Tools import listToStr as concat

import numpy
from numpy import linalg

# variables
__file = __file__.replace('\\', '/') # windows
CURRENTDIR = os.path.dirname(os.path.abspath(__file__))+'/'
CURRENTDIR = CURRENTDIR.replace('//', '/') # windows compatible filename

#=====================================================================================
# Scene lauch
#=====================================================================================
def createScene(root_node):
    root_node.createObject('RequiredPlugin', name='image')
    root_node.createObject('RequiredPlugin', name='Flexible')
    root_node.createObject('RequiredPlugin', name='Compliant')
    root_node.createObject('RequiredPlugin', name='Registration')

    root_node.createObject('CompliantAttachButtonSetting')

    root_node.createObject('PythonScriptController', name='MyClass', filename=__file, classname='MyClass')

# ================================================================= #
# Creation of the scene
# ================================================================= #
class MyClass(Sofa.PythonScriptController):

     # Setup of class attributes
    def setup(self):
        return

    def createGraph(self, root):

        self.setup()

        self.node = root

        self.node.createObject('VisualStyle', displayFlags='showVisual hideWireframe showBehaviorModels showForceFields showInteractionForceFields')
        self.node.createObject('BackgroundSetting',color='1 1 1')
        self.node.gravity = '0 -9.81 0'
        self.node.dt = .01

        # compliant solver
        self.node.createObject('CompliantImplicitSolver', stabilization=1)
        self.node.createObject('SequentialSolver', iterations=75, precision=1E-15, iterateOnBilaterals=1)
        self.node.createObject('LDLTResponse', schur=0)

        # beam creation
        self.mainNode = self.node.createChild('deformable')
        self.mainNode.createObject('RegularGrid', name='grid', n='25 5 5', min='0. 0. 0.', max='4. 1. 1.')
        self.mainNode.createObject('MeshToImageEngine', template='ImageUC', name='rasterizer', src='@grid', value=1, insideValue=1, voxelSize=0.025, padSize=0, rotateImage='false')
        self.mainNode.createObject('ImageContainer', template='ImageUC', name='image', src='@rasterizer', drawBB='false')
        self.mainNode.createObject('ImageSampler', template='ImageUC', name='sampler', src='@image', method=1, param='2 0', clearData=0)
        self.mainNode.createObject('MechanicalObject', template='Affine', name='parent', position='@sampler.position', rest_position='@sampler.position', showObject=1, showObjectScale='0.1')
        self.mainNode.createObject('VoronoiShapeFunction', template='ShapeFunctiond,ImageUC', name='SF', position='@parent.rest_position', image='@image.image', transform='@image.transform', nbRef=4, clearData=1, bias=0)
        self.mainNode.createObject('FixedConstraint', template='Affine', indices='0')

        # behavior
        behaviorNode = self.mainNode.createChild('behavior')
        behaviorNode.createObject('ImageGaussPointSampler', name='sampler', indices='@../SF.indices', weights='@../SF.weights', transform='@../SF.transform', method=2, order=4, targetNumber=10)
        behaviorNode.createObject('MechanicalObject', template='F332')
        behaviorNode.createObject('LinearMapping', template='Affine,F332')
        eNode = behaviorNode.createChild('E')
        eNode.createObject('MechanicalObject', template='E332', name='E')
        eNode.createObject('CorotationalStrainMapping', template='F332,E332', method='polar')
        eNode.createObject('HookeForceField', template='E332', name='ff', youngModulus='1E3', poissonRatio='0', viscosity='0')

        # contact and visual model
        contactNode = self.mainNode.createChild('registration')
        contactNode.createObject('MeshTopology', name='topo', src='@../grid')
        contactNode.createObject('MechanicalObject', name='DOFs')
        contactNode.createObject('UniformMass', totalMass=1)
        contactNode.createObject('TriangleModel')
        contactNode.createObject('LinearMapping', template='Affine,Vec3d')

        visuNode = contactNode.createChild('visual')
        visuNode.createObject('OglModel', template='ExtVec3f', name='visual', src='@../topo', color='0.8 0.2 0.2 1')
        visuNode.createObject('IdentityMapping', template='Vec3d,ExtVec3f')

        global sceneDataIO
        sceneDataIO = SofaPython.Tools.SceneDataIO(self.node)
        sceneDataIO.classNameList = ['MechanicalObject', 'OglModel', 'VisualModel']

    def bwdInitGraph(self, root):
        print 'bwdInitGraph: backward init'
        self.loadState('./SceneDataIO/')

    # ===============================================================================
    # Scene methods
    # ===============================================================================
    def loadState(self, directory=None):
        state = sceneDataIO.readData(directory)
