import Sofa

from Compliant import StructuralAPI, Tools

StructuralAPI.geometric_stiffness = 2

path = Tools.path( __file__ )

def createFixedRigidBody(node,name,pos):
    body = StructuralAPI.RigidBody( node, name )
    body.setManually( [pos,0,0,0,0,0,1], 1, [1,1,1] )
    body.node.createObject('FixedConstraint')
    return body
  
def createRigidBody(node,name,posx,posy=0):
    body = StructuralAPI.RigidBody( node, name )
    body.setManually( [posx,posy,0,0,0,0,1], 1, [1,1,1] )
    # enforce rigid visualization
    body.dofs.showObject=True
    body.dofs.showObjectScale=1
    # add initial velocity to make it move!
    body.dofs.velocity = "10 10 10 10 10 10"
    # add a collision model to be able to rotate it with the mouse
    collisionModel = body.addCollisionMesh( "mesh/cube.obj" )
    return body
    
    
    
def createScene(root):
  
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehavior showWireframe showCollisionModels" )
    root.dt = 0.01
    root.gravity = [0, -10, 0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('CompliantAttachButtonSetting')
    
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=0, neglecting_compliance_forces_in_geometric_stiffness=0)
    root.createObject('SequentialSolver', iterations=100, precision=0)
    #root.createObject('LUResponse')
    root.createObject('LDLTResponse')
    
    
    bodies = []
    points = []
    
    N = 10
   
    for i in xrange(N):
        body = StructuralAPI.RigidBody( root, "body_"+str(i) )
        body.setManually( [i,0,0,0,0,0,1], 1, [1,1,1] )
        body.dofs.showObject = True
        body.dofs.showObjectScale = .5
        bodies.append( body )
        
    bodies[0].node.createObject('FixedConstraint')
    bodies[N-1].mass.mass = 10
    bodies[N-1].mass.inertia = "10 10 10"
    
        
    for i in xrange(N-1):
        p0 = bodies[i].addMappedPoint( "right", [0.5, 0, 0] )
        p0.dofs.showObject = True
        p0.dofs.showObjectScale = .1
        p0.dofs.drawMode=1
        p1 = bodies[i+1].addMappedPoint( "left", [-0.5, 0, 0] )
        p1.dofs.showObject = True
        p1.dofs.showObjectScale = .1
        p1.dofs.drawMode=2
        d = p0.node.createChild( "d"+str(i) )
        d.createObject('MechanicalObject', template = 'Vec3'+StructuralAPI.template_suffix, name = 'dofs', position = '0 0 0' )
        input = [] # @internal
        input.append( '@' + Tools.node_path_rel(root,p0.node) + '/dofs' )
        input.append( '@' + Tools.node_path_rel(root,p1.node) + '/dofs' )
        d.createObject('DifferenceMultiMapping', name = 'mapping', input = Tools.cat(input), output = '@dofs', pairs = "0 0" )
        p1.node.addChild( d )
        d.createObject('UniformCompliance', name = 'compliance', compliance="0" )
    
