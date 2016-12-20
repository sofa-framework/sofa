import Sofa

from Compliant import StructuralAPI
from SofaPython.Tools import listToStr as concat

StructuralAPI.geometric_stiffness=2

def createScene(root):
    
    # root node setup
    root.createObject('RequiredPlugin', pluginName = 'Flexible')
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    root.createObject('CompliantAttachButtonSetting' )
    root.createObject('BackgroundSetting', color='1 1 1')

    # simuation parameters
    root.dt = 1e-1
    root.gravity = [0, 0, 0]
    
    # ode solver
    ode = root.createObject('CompliantImplicitSolver', neglecting_compliance_forces_in_geometric_stiffness=False, stabilization = "pre-stabilization")
    
    # numerical solver
    # root.createObject('LDLTSolver', name="numsolver")
    root.createObject('SequentialSolver', name='numsolver', iterations=250, precision=1e-14, iterateOnBilaterals=True)
    # root.createObject('LDLTResponse', name='response')
    # root.createObject('LUResponse', name='response')
    
    # scene node
    scene = root.createChild('scene')
    
    # script variables
    nbLink = 7
    linkSize = 2

    # links creation
    links = []

    # rigid bodies
    for i in xrange(nbLink):
        body = StructuralAPI.RigidBody(root, "link-{0}".format(i))
        body.setManually(offset = [0, -1.*linkSize * i, 0, 0,0,0,1], inertia_forces = False)
        body.dofs.showObject = True
        body.dofs.showObjectScale = 0.25*linkSize
        body.addVisualModel("mesh/cylinder.obj", scale3d=[.3,.1*linkSize,.3], offset=[0,-0.5*linkSize,0,0,0,0,1])
        links.append( body )

    # attach first link
    links[0].setFixed()
    
    # joints creation
    points = []
    for i in xrange( nbLink-1 ):
        off1 = links[i].addOffset("offset-{0}-{1}".format(i, i+1), [0, -0.5*linkSize, 0, 0,0,0,1])
        off2 = links[i+1].addOffset("offset-{0}-{1}".format(i+1, i), [0, 0.5*linkSize, 0, 0,0,0,1])
        j = StructuralAPI.HingeRigidJoint(2, "joint-{0}-{1}".format(i, i+1), off1.node, off2.node, isCompliance=False, compliance=1E-5)
        #StructuralAPI.BallAndSocketRigidJoint("joint-{0}-{1}".format(i, i+1), off1.node, off2.node, isCompliance=True, compliance=0)
        j.addLimits(0,.8,compliance=1E-15)
        points.append( links[i].addAbsoluteMappedPoint("point-{0}-{1}".format(i, i+1), [0.5, -1.*linkSize*i -0.25*linkSize , 0]) )
        points.append( links[i+1].addAbsoluteMappedPoint("point-{0}-{1}".format(i+1, i), [0.5, -1.*linkSize*(i+1) +0.25*linkSize , 0]) )

    points.append( links[nbLink-1].addAbsoluteMappedPoint("point-{0}".format(nbLink), [0.5, -1.*linkSize*(nbLink-1)-0.5*linkSize, 0]) )

    # rod
    rodNode = scene.createChild('rod')
    rodNode.createObject( 'MechanicalObject', name='dofs', template="Vec3", position=concat([0.5, 0.5*linkSize ,0]) )
    rodNode.createObject('UniformMass',  name="mass" ,totalMass="1")
    rodNode.createObject('RestShapeSpringsForceField',points="0", stiffness="1E3")

    fullrodNode = rodNode.createChild('fullrod')
    input='@'+rodNode.getPathName()+' '
    indexPairs='0 0 '
    edges=''
    for i,p in enumerate(points):
        input+='@'+p.node.getPathName()+' '
        indexPairs+=str(i+1)+' 0 '
        edges+=str(i)+' '+str(i+1)+' '
        p.node.addChild(fullrodNode)

    fullrodNode.createObject('MechanicalObject', template="Vec3", showObject=True, showObjectScale=0.1, showColor="0 1 0 1", drawMode=1)
    fullrodNode.createObject('SubsetMultiMapping', template = "Vec3,Vec3", name="mapping", input = input , output = '@./', indexPairs=indexPairs)

    Lnode = fullrodNode.createChild("L")
    Lnode.createObject('MechanicalObject', template="Vec1" )
    Lnode.createObject('LengthMapping', template="Vec3,Vec1", edges=edges, offset=str(-nbLink*linkSize) , geometricStiffness=1 )
    Lnode.createObject('UniformCompliance', compliance=1E-10,rayleighStiffness=0,isCompliance='1')

    Vnode = fullrodNode.createChild("visual")
    Vnode.createObject('VisualModel',edges=edges)
    Vnode.createObject('IdentityMapping')

