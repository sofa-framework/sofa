import Sofa
import SofaPython.Tools
import Compliant.StructuralAPI
import RigidScale.API

def createScene(node):
    
    node.createObject('VisualStyle', displayFlags='showBehaviorModels hideCollisionModels hideMappings hideForceFields')
    
    node.dt=0.01
    node.gravity='0 0 0'
    
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

    # scene        
    sceneNode = node.createChild('scene') 
    
    body = RigidScale.API.ShearlessAffineBody(sceneNode, "Roue")
    mesh = SofaPython.Tools.localPath(__file__, "3.obj")
    body.setFromMesh(mesh, voxelSize=0.05, numberOfPoints=5, offset=[1.,0,0,0,0,0,1])
    body.image.addViewer()
    body.addBehavior(youngModulus=1e4, numberOfGaussPoint=50)
    
    cm = body.addCollisionMesh(mesh)
    cm.addVisualModel() # visual model similar to collision model
    
    body.affineDofs.showObject=True
    body.affineDofs.showObjectScale=0.5
    
    offset0 = body.addOffset("offset0", [0, 0, 0.148, 0, 0, 0, 1])
    offset0.dofs.showObject=True
    offset0.dofs.showObjectScale=0.25
    
    offset1 = body.addOffset("offset1", [0.24, -0.145, 0.478, 0, 0, 0, 1])
    offset1.dofs.showObject=True
    offset1.dofs.showObjectScale=0.25
    
    body.setFixed(True)