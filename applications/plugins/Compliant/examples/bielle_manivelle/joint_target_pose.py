import Sofa

from Compliant import Rigid, Tools

mesh_path = Tools.path( __file__ )

scale = 1

# parts of the mechanism
parts = [ 
    ["Corps","Corps.msh","1.36 0 0.0268 0 0 0 1","0 0 0 0 0 0 1","22.8 751 737", "2.1e+11","0.28","7.8e+3",1291.453/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","0.obj","Actor_Sensor_NA",],
    ["Roue","Roue.msh","0 -0.00604 0.354 0 0 0 1","0 0 -0.148 0 0 0 1","105 106 205", "2.1e+11","0.28","7.8e+3",780.336/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","3.obj","Actor_Sensor_NA"],
   ["Came","Came.msh","0 0 -0.00768 0 0 0 1","1.085 -0.072 0.33 0 0 0 1","40.5 40.6 0.331", "2.1e+11","0.28","7.8e+3",161.416/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","2.obj","Actor_Sensor_NA"],
   ["Piston","Piston.msh","0 0 0.424 0 0 0 1","2.05 0 0.33 0 0 0 1","0.356 14.6 14.7", "2.1e+11","0.28","7.8e+3",132.759/scale,"TetrahedronFEMForceField","Rigid","Vec3d","TLineModel","TPointModel","ExtVec3f","1.obj","Actor_Sensor_NA"]
]


# joint offsets
offset = [
    [0, Rigid.Frame().read('0 0 0 0 0 0 1')],
    [1, Rigid.Frame().read('0 0 0.148 0 0 0 1')],

    [1, Rigid.Frame().read('0.24 -0.145 0.478 0 0 0 1')],
    [2, Rigid.Frame().read('-0.845 -0.073 0 0 0 0 1')],

    [2, Rigid.Frame().read('0.852 0.072 0 0 0 0 1')],
    [3, Rigid.Frame().read('-0.113 0 0 0 0 0 1')],

    [3, Rigid.Frame().read('0.15 0 0 0 0 0 1')],
    [0, Rigid.Frame().read('2.2 0 0.33 0 0 0 1')]
]

# joints: parent offset, child offset, joint def
links = [ 

    # revolute joint around z
    [0, 1, Rigid.RevoluteJoint(2), [0,0,0,0,0,1.7]], # corps-roue
    [2, 3, Rigid.RevoluteJoint(2)], # roue-came
    [4, 5, Rigid.RevoluteJoint(2)], # came-piston

    # sliding joint around x
    [6, 7, Rigid.PrismaticJoint(0)]
]
    



def createScene(node):
    node.createObject('RequiredPlugin', pluginName = 'Compliant' )

    node.createObject('VisualStyle', displayFlags='hideBehaviorModels hideCollisionModels hideMappings hideForceFields')
    node.createObject('Stabilization', name='Group')
    node.findData('dt').value=0.01
    
    node.findData('gravity').value='0 -9.81 0'
    node.createObject('CompliantImplicitSolver',
                      name='odesolver', 
                      stabilization="true")
        
    node.createObject('MinresSolver',
                      name = 'numsolver',
                      iterations = '250',
                      precision = '1e-14')
        
    scene = node.createChild('scene') 
        
    rigid = []
    joint = []
        
    # create rigid bodies
    for p in parts:
        r = Rigid.Body()
        r.name = p[0]
        
        # r.collision = part_path + p[1]
        r.dofs.read( p[3] )
        r.visual = mesh_path + '/' + p[15]
        r.collision = r.visual
        r.inertia_forces = True
        
        density = float(p[7])
        r.mass_from_mesh( r.visual, density )

        r.insert( scene )
        rigid.append( r )
        
    # create joints
    for i in links:
        j = i[2]
        j.compliance = 0
        j.damping = 5
        
        p = offset[i[0]][0]
        off_p = offset[i[0]][1]
        
        c = offset[i[1]][0]
        off_c = offset[i[1]][1]
        
        j.append(rigid[p].user, off_p)
        j.append(rigid[c].user, off_c)
        
        joint.append( j.insert( scene) )
        if len(i)==4 :
            j.setTargetPose(i[3], compliance=1e-5, damping=10)
        
    # fix first body
    rigid[0].node.createObject('FixedConstraint', indices = '0' )
            
    return node

