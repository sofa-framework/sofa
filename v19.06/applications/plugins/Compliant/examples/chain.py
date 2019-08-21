import Sofa

from Compliant import StructuralAPI

StructuralAPI.geometric_stiffness=2

def createScene(root):
    
    # root node setup
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    root.createObject('CompliantAttachButtonSetting' )
    
    # simuation parameters
    root.dt = 1e-2
    root.gravity = [0, -9.8, 0]
    
    # ode solver
    ode = root.createObject('CompliantImplicitSolver', neglecting_compliance_forces_in_geometric_stiffness=False, stabilization = "pre-stabilization")
    
    # numerical solver
    root.createObject('SequentialSolver', name='numsolver', iterations=250, precision=1e-14, iterateOnBilaterals=True)
    root.createObject('LDLTResponse', name='response')
    #root.createObject('LUResponse', name='response')
    
    # scene node
    scene = root.createChild('scene')
    
    # script variables
    nbLink = 10
    linkSize = 2
    
    # links creation
    links = []
    
    # rigid bodies
    for i in xrange(nbLink):
        body = StructuralAPI.RigidBody(root, "link-{0}".format(i))
        body.setManually(offset = [0, -1.*linkSize * i, 0, 0,0,0,1], inertia_forces = True)
        body.dofs.showObject = True
        body.dofs.showObjectScale = 0.25*linkSize
        links.append( body )
    # attach first link
    links[0].setFixed()
    
    # joints creation
    for i in xrange( nbLink-1 ):
        off1 = links[i].addOffset("offset-{0}-{1}".format(i, i+1), [0, -0.5*linkSize, 0, 0,0,0,1])
        off2 = links[i+1].addOffset("offset-{0}-{1}".format(i+1, i), [0, 0.5*linkSize, 0, 0,0,0,1])
        StructuralAPI.HingeRigidJoint(2, "joint-{0}-{1}".format(i, i+1), off1.node, off2.node, isCompliance=True, compliance=0)
        #StructuralAPI.BallAndSocketRigidJoint("joint-{0}-{1}".format(i, i+1), off1.node, off2.node, isCompliance=True, compliance=0)
        
        
