import os.path

import Sofa

import Compliant.sml

def createScene(node):
    
    node.dt=0.01
    node.gravity='0 0 0'
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('CompliantAttachButtonSetting' )
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    node.createObject('MinresSolver', name='numsolver', iterations='250', precision='1e-14');
    
    scene_two_bones = Compliant.sml.Scene(os.path.join(os.path.dirname(__file__),"two_bones.xml"))
    scene_two_bones.param.showRigid=True
    scene_two_bones.param.showOffset=True
    scene_two_bones.createScene(node)
    
    scene_two_bones.rigids["bone01"].node.createObject('FixedConstraint')
    
    scene_two_bones.deformable["skin"].dofs.showObject=True
    scene_two_bones.deformable["skin"].dofs.drawMode=1
    
    return node
