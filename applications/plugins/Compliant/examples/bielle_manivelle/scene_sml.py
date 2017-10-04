import Sofa
import SofaPython

import SofaPython.units
import SofaPython.sml
import SofaPython.SofaNumpy
import Compliant.sml

def createScene(node):
    
    node.dt=0.01
    node.gravity=[0, -9.81, 0]
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('CompliantAttachButtonSetting' )
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    node.createObject('MinresSolver', name='numsolver', iterations='250', precision='1e-14')


    model = SofaPython.sml.Model( SofaPython.Tools.localPath( __file__, "bielle_manivelle.sml") )
    
    scene_bielle_manivelle = Compliant.sml.SceneArticulatedRigid(node, model)
    scene_bielle_manivelle.material.load( SofaPython.Tools.localPath( __file__, "material.json") )
    scene_bielle_manivelle.setMaterialByTag("part", "steel")
    
    scene_bielle_manivelle.param.simuLengthUnit="dm"
    scene_bielle_manivelle.param.showRigid=True
    scene_bielle_manivelle.param.showOffset=True
    scene_bielle_manivelle.createScene()
    
    scene_bielle_manivelle.rigids["1"].node.createObject('FixedConstraint')
    
    gravity = SofaPython.SofaNumpy.numpy_data(node, "gravity")
    gravity[0] = SofaPython.units.acceleration_from_SI(gravity[0])
    
    return node

