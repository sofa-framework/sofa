import os.path

import Sofa

import SofaPython.sml
import Compliant.sml
import RigidScale.sml


def createScene(node):
    
    node.dt = 0.01
    node.gravity = '0 0 0'
    
    node.createObject('RequiredPlugin', pluginName='image')
    node.createObject('RequiredPlugin', pluginName='Flexible')
    node.createObject('RequiredPlugin', pluginName='Compliant')
    node.createObject('RequiredPlugin', pluginName='RigidScale')
    
    node.createObject('CompliantAttachButtonSetting')
    node.createObject('CompliantImplicitSolver', name='odesolver')
    node.createObject('SequentialSolver', name='seqsolver', precision="1e-10", iterations="500")
    
    model = SofaPython.sml.Model(os.path.join(os.path.dirname(__file__), "skinning.sml"))
    scene = RigidScale.sml.SceneSkinningRigidScale(node, model)
    scene.param.showAffine = True
    scene.param.showAffineScale = 0.5
    scene.param.showOffset = True
    
    scene.createScene()
    
    return node
