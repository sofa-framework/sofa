import os.path

import Sofa

import SofaPython.sml
import Compliant.sml

def createScene(node):
    
    node.dt=0.01
    node.gravity='0 -9.81 0'
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('CompliantAttachButtonSetting' )
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    node.createObject('MinresSolver', name='numsolver', iterations='250', precision='1e-14');
    
    model = SofaPython.sml.Model(os.path.join(os.path.dirname(__file__), "bielle_manivelle.xml"))
    
    scene_bielle_manivelle = Compliant.sml.SceneArticulatedRigid(node, model)
    scene_bielle_manivelle.material.load(os.path.join(os.path.dirname(__file__), "material.json"))
    scene_bielle_manivelle.param.showRigid=True
    scene_bielle_manivelle.param.showOffset=True
    scene_bielle_manivelle.createScene()
    
    scene_bielle_manivelle.rigids["1"].node.createObject('FixedConstraint')
    
    return node

