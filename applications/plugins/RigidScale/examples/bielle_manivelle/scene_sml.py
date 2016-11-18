import Sofa
import SofaPython.sml
import RigidScale.sml

def createScene(node):
    
    model = SofaPython.sml.Model("bielle_manivelle.sml")
    scene = RigidScale.sml.SceneArticulatedRigidScale(node, model)
    
    scene.param.voxelSize = 0.005
    scene.param.showAffine=True
    scene.param.showOffset=True
    scene.param.showImage=False
    scene.param.rigidScaleNbDofByTag["bigDeformation"]=1
    
    scene.createScene()
    
    scene.rigidScales["1"].setFixed(True)
    
    node.createObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels')
    
    node.dt=0.01
    node.gravity='0 -9.81 0'
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('RequiredPlugin', name = 'Flexible' )
    node.createObject('RequiredPlugin', name = 'image' )
    node.createObject('RequiredPlugin', name = 'RigidScale' )
    node.createObject('CompliantAttachButtonSetting' )
    
    # solvers
    compliance = 0    
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    node.createObject('SequentialSolver', iterations=25, precision=1E-15, iterateOnBilaterals=1)
    node.createObject('LDLTResponse', schur=0)