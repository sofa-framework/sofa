import Sofa
import SofaPython.Tools
import Compliant.StructuralAPI
import RigidScale.API

# parts of the mechanism
parts = [ 
    ["Corps",[0, 0, 0, 0, 0, 0, 1],"7.8e+3","0.obj"],
    ["Roue",[0, 0, -0.148, 0, 0, 0, 1],"7.8e+3","3.obj"],
    ["Came",[1.085, -0.072, 0.33, 0, 0, 0, 1],"7.8e+3","2.obj"],
    ["Piston",[2.05, 0, 0.33, 0, 0, 0, 1],"7.8e+3","1.obj"]
]


# joint offsets: name, rigid body, offset
joint_offsets = [
    ["offset0", 0, [0, 0, 0, 0, 0, 0, 1]],
    ["offset0", 1, [0, 0, 0.148, 0, 0, 0, 1]],

    ["offset1", 1, [0.24, -0.145, 0.478, 0, 0, 0, 1]],
    ["offset1", 2, [-0.845, -0.073, 0, 0, 0, 0, 1]],

    ["offset2", 2, [0.852, 0.072, 0, 0, 0, 0, 1]],
    ["offset2", 3, [-0.113, 0, 0, 0, 0, 0, 1]],

    ["offset3", 3, [0.15, 0, 0, 0, 0, 0, 1]],
    ["offset3", 0, [2.2, 0, 0.33, 0, 0, 0, 1]]    
]

# joints: name, offset1, offset2, joint type, joint axis
links = [ 

    #revolute joint around z
    ["hinge_corps-roue",    0, 1, Compliant.StructuralAPI.HingeRigidJoint,  2],
    ["hinge_roue-came",     2, 3, Compliant.StructuralAPI.HingeRigidJoint,  2],
    ["hinge_came-piston",   4, 5, Compliant.StructuralAPI.HingeRigidJoint,  2],

    # sliding joint around x
    ["slider_corps-piston", 6, 7, Compliant.StructuralAPI.SliderRigidJoint, 0]
]
    



def createScene(node):
    
    node.createObject('VisualStyle', displayFlags='showBehaviorModels hideCollisionModels hideMappings hideForceFields')
    
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
    #node.createObject('MinresSolver', name='numsolver', iterations='500', precision='1e-14');
    node.createObject('LDLTSolver', name='numsolver', schur=0)

    # scene        
    sceneNode = node.createChild('scene') 
    bodies  = []
    offsets = []
    joints  = []
        
    # create ShearlessAffineBody
    for p in parts:
        body = RigidScale.API.ShearlessAffineBody( sceneNode, p[0] )
        
        mesh = SofaPython.Tools.localPath(__file__, p[3])
        density = float(p[2])
        offset = p[1]
        body.setFromMesh( mesh, density, offset, voxelSize=0.1 )
        
        cm = body.addCollisionMesh( mesh )
        cm.addVisualModel() # visual model similar to collision model
        
        body.affineDofs.showObject=True
        body.affineDofs.showObjectScale=0.5
        
        bodies.append(body)
      
    # fix first body
    bodies[0].setFixed(True)
      
    # create offsets
    for o in joint_offsets:
      
        o = bodies[o[1]].addRigidScaleOffset( o[0], o[2] )
        
        o.dofs.showObject=True
        o.dofs.showObjectScale=0.25
        
        offsets.append( o )

    ## create joints
    for l in links:
      
      j = l[3] (l[4], l[0], offsets[l[1]].node, offsets[l[2]].node )
      j.constraint.compliance.compliance = compliance
      
      joints.append( j )
          
    # just for fun!
    #rigids[1].addMotor([0,0,0,0,0,1000])
            
    return node

