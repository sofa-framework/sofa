import os.path

import Sofa

import SofaPython.sml
import Compliant.sml
import RigidScale.sml


def createScene(node):
    
    node.dt=0.01
    node.gravity='0 0 0'
    
    node.createObject('RequiredPlugin', pluginName='image')
    node.createObject('RequiredPlugin', pluginName='Flexible')
    node.createObject('RequiredPlugin', pluginName='Compliant')
    node.createObject('RequiredPlugin', pluginName='RigidScale')
    
    node.createObject('CompliantAttachButtonSetting' )
    node.createObject('CompliantImplicitSolver', name='odesolver')
    node.createObject('SequentialSolver', name='seqsolver',  precision="1e-10", iterations="500")

    node.createObject('VisualStyle', displayFlags="showVisual showBehavior showWireframe hideCollisionModels" )
    
    model = SofaPython.sml.Model(os.path.join(os.path.dirname(__file__),"two_bones.xml"))
    scene_two_bones = RigidScale.sml.SceneSkinningRigidScale(node, model)

    scene_two_bones.param.showRigid = False
    scene_two_bones.param.showRigidScale = 0.1

    scene_two_bones.param.showAffine = True
    scene_two_bones.param.showAffineScale = 0.3

    scene_two_bones.param.showOffset = True
    scene_two_bones.param.showOffsetScale = 0.1

    scene_two_bones.createScene()
    
    scene_two_bones.rigidScales["bone01"].rigidNode.createObject('FixedConstraint')
    scene_two_bones.rigidScales["bone01"].scaleNode.createObject('FixedConstraint')

    scene_two_bones.deformables["skin"].dofs.showObject = True
    scene_two_bones.deformables["skin"].dofs.drawMode = 1
    
    return node
