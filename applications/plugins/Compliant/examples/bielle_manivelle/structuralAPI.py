import Sofa

from Compliant import StructuralAPI, Tools

mesh_path = Tools.path( __file__ )

scale = 1

# parts of the mechanism
parts = [ 
    ["Corps","Corps.msh","1.36 0 0.0268 0 0 0 1",[0, 0, 0, 0, 0, 0, 1],"22.8 751 737", "2.1e+11","0.28","7.8e+3",1291.453/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","0.obj","Actor_Sensor_NA",],
    ["Roue","Roue.msh","0 -0.00604 0.354 0 0 0 1",[0, 0, -0.148, 0, 0, 0, 1],"105 106 205", "2.1e+11","0.28","7.8e+3",780.336/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","3.obj","Actor_Sensor_NA"],
    ["Came","Came.msh","0 0 -0.00768 0 0 0 1",[1.085, -0.072, 0.33, 0, 0, 0, 1],"40.5 40.6 0.331", "2.1e+11","0.28","7.8e+3",161.416/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","2.obj","Actor_Sensor_NA"],
    ["Piston","Piston.msh","0 0 0.424 0 0 0 1",[2.05, 0, 0.33, 0, 0, 0, 1],"0.356 14.6 14.7", "2.1e+11","0.28","7.8e+3",132.759/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","1.obj","Actor_Sensor_NA"]
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
    node.createObject('MinresSolver', name='numsolver', iterations='250', precision='1e-14');
    #node.createObject('LDLTSolver', name='numsolver'); compliance = 1e-10 #need to relax the system a bit
        
        
        
    # scene    
        
    scene = node.createChild('scene') 
      
    rigids  = []
    offsets = []
    joints  = []
    
    
    
        
    # create rigid bodies
    for p in parts:
      
        r = StructuralAPI.RigidBody( scene, p[0] )
        
        mesh = mesh_path + '/' + p[15]
        density = float(p[7])
        offset = p[3]
        
        r.setFromMesh( mesh, density, offset )
        
        r.addCollisionMesh( mesh )
        r.addVisualModel( mesh )
        
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

