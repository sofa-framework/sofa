import os.path

import Sofa

import Compliant.sml

def createScene(node):
    
    node.dt=0.01
    node.gravity='0 -9.81 0'
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('CompliantAttachButtonSetting' )
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    node.createObject('MinresSolver', name='numsolver', iterations='250', precision='1e-14');
    
    scene_bielle_manivelle = Compliant.sml.Scene(os.path.join(os.path.dirname(__file__),"bielle_manivelle.xml"))
    scene_bielle_manivelle.param.showRigid=True
    scene_bielle_manivelle.param.showOffset=True
    scene_bielle_manivelle.createScene(node)
    
    scene_bielle_manivelle.rigids["1"].node.createObject('FixedConstraint')
    
    return node

