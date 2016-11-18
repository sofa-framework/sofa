import os.path

import Sofa

import SofaPython.sml
import Compliant.sml

def createScene(node):
    
    node.dt=0.01
    node.gravity='0 0 0'
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('CompliantAttachButtonSetting' )
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    node.createObject('MinresSolver', name='numsolver', iterations='250', precision='1e-14');
    
    model = SofaPython.sml.Model(os.path.join(os.path.dirname(__file__),"two_bones.xml"))
    scene_two_bones = Compliant.sml.SceneSkinning(node, model)
    scene_two_bones.param.showRigid = True
    scene_two_bones.param.showOffset = True
    scene_two_bones.param.showRigidScale = 0.05
    scene_two_bones.param.showOffsetScale = 0.025
    
    scene_two_bones.createScene()
    
    scene_two_bones.rigids["bone01"].node.createObject('FixedConstraint')
    scene_two_bones.deformables["skin"].dofs.showObject=True
    scene_two_bones.deformables["skin"].dofs.drawMode=1
    
    return node
