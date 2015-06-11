import Sofa

from Compliant import StructuralAPI, Tools

mesh_path = Tools.path( __file__ )

scale = 1

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
    ["hinge_corps-roue",    0, 1, StructuralAPI.HingeRigidJoint,  2],
    ["hinge_roue-came",     2, 3, StructuralAPI.HingeRigidJoint,  2],
    ["hinge_came-piston",   4, 5, StructuralAPI.HingeRigidJoint,  2],

    # sliding joint around x
    ["slider_corps-piston", 6, 7, StructuralAPI.SliderRigidJoint, 0]
]
    



def createScene(node):
  
  
    # global
    
    node.createObject('VisualStyle', displayFlags='hideBehaviorModels hideCollisionModels hideMappings hideForceFields')
    
    node.dt=0.01
    node.gravity='0 -9.81 0'
    
    node.createObject('RequiredPlugin', name = 'Compliant' )
    node.createObject('CompliantAttachButtonSetting' )

    
    
    # solvers
    compliance = 0
    
    node.createObject('CompliantImplicitSolver', name='odesolver',stabilization=1)
    #node.createObject('MinresSolver', name='numsolver', iterations='500', precision='1e-14');
    node.createObject('LDLTSolver', name='numsolver', schur=0)
        
        
    # scene    
        
    scene = node.createChild('scene') 
      
    rigids  = []
    offsets = []
    joints  = []
    
    
    
        
    # create rigid bodies
    for p in parts:
      
        r = StructuralAPI.RigidBody( scene, p[0] )
        
        mesh = mesh_path + '/' + p[3]
        density = float(p[2])
        offset = p[1]
        
        r.setFromMesh( mesh, density, offset )
        
        cm = r.addCollisionMesh( mesh )
        cm.addVisualModel() # visual model similar to collision model
        #r.addVisualModel( mesh ) # if the visual model was different from the collision model
        
        r.dofs.showObject=True
        r.dofs.showObjectScale=0.5
        
        rigids.append(r)
      
      
    # fix first body
    rigids[0].node.createObject('FixedConstraint')
    
   
    
      
    # create offsets
    for o in joint_offsets:
      
        o = rigids[o[1]].addOffset( o[0], o[2] )
        
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

