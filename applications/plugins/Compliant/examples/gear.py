import Sofa

from Compliant import StructuralAPI, Tools

StructuralAPI.geometric_stiffness = 2

path = Tools.path( __file__ )

    
    
    
def createScene(root):
  
    ##### global parameters
    root.createObject('VisualStyle', displayFlags="showBehavior showCollisionModels" )
    root.dt = 0.01
    root.gravity = [0, -10, 0]
    
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('CompliantAttachButton')
    
    
    root.createObject('DefaultPipeline', name='DefaultCollisionPipeline', depth="6")
    root.createObject('BruteForceDetection')
    root.createObject('DiscreteIntersection')
    root.createObject('DefaultContactManager', name="Response", response="CompliantContact", responseParams="compliance=0&restitution=0" )
    
    
    
    
    
    ##### SOLVER
    root.createObject('CompliantImplicitSolver', stabilization=1, neglecting_compliance_forces_in_geometric_stiffness=1)
    root.createObject('SequentialSolver', iterations=100, precision=0)
    #root.createObject('LUResponse')
    root.createObject('LDLTResponse')
    
    
    
    
    
    ##### GEAR
    
    gearNode = root.createChild( "GEAR" )
    
    r0 = 0.33
    r1 = 0.66
    
    body0 = StructuralAPI.RigidBody( gearNode, "body_0" )
    body0.setManually( [0,0,0,0,0,0,1], 1, [1,1,1] )
    body0.dofs.showObject = True
    body0.dofs.showObjectScale = r0*1.1
    body0.dofs.velocity="0 0 0 0 1 0"
    body0.node.createObject('Sphere', radius=r0)
        
    body1 = StructuralAPI.RigidBody( gearNode, "body_1" )
    body1.setManually( [1,0,0,0,0,0,1], 1, [1,1,1] )
    body1.dofs.showObject = True
    body1.dofs.showObjectScale = r1*1.1
    body1.node.createObject('Sphere', radius=r1)
    
   
    body0.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 1 0 1")
    body1.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 1 0 1")
    
    
    d = body0.node.createChild( "d" )
    d.createObject('MechanicalObject', template = 'Vec1'+StructuralAPI.template_suffix, name = 'dofs', position = '0 0 0' )
    input = [] # @internal
    input.append( '@' + Tools.node_path_rel(root,body0.node) + '/dofs' )
    input.append( '@' + Tools.node_path_rel(root,body1.node) + '/dofs' )
    d.createObject('GearMultiMapping', name = 'mapping', input = Tools.cat(input), output = '@dofs', pairs = "0 4 0 4", ratio = -r0/r1 )
    body1.node.addChild( d )
    d.createObject('UniformCompliance', name = 'compliance', compliance="0" )
    
    
    
    
    #####  driving belt / chain
    
    
    beltNode = root.createChild( "BELT" )
    
    r0 = 0.7
    r1 = 0.3
    
    body0 = StructuralAPI.RigidBody( beltNode, "body_0" )
    body0.setManually( [0,-2,0,0,0,0,1], 1, [1,1,1] )
    body0.dofs.showObject = True
    body0.dofs.showObjectScale = r0*1.1
    body0.dofs.velocity="0 0 0 0 1 0"
    body0.node.createObject('Sphere', radius=r0)
        
    body1 = StructuralAPI.RigidBody( beltNode, "body_1" )
    body1.setManually( [1.5,-2,0,0,0,0,1], 1, [1,1,1] )
    body1.dofs.showObject = True
    body1.dofs.showObjectScale = r1*1.1
    body1.node.createObject('Sphere', radius=r1)
    
   
    body0.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 1 0 1")
    body1.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 1 0 1")
    
    
    d = body0.node.createChild( "d" )
    d.createObject('MechanicalObject', template = 'Vec1'+StructuralAPI.template_suffix, name = 'dofs', position = '0 0 0' )
    input = [] # @internal
    input.append( '@' + Tools.node_path_rel(root,body0.node) + '/dofs' )
    input.append( '@' + Tools.node_path_rel(root,body1.node) + '/dofs' )
    d.createObject('GearMultiMapping', name = 'mapping', input = Tools.cat(input), output = '@dofs', pairs = "0 4 0 4", ratio = r0/r1 )
    body1.node.addChild( d )
    d.createObject('UniformCompliance', name = 'compliance', compliance="0" )
    
    
    
    
    
    
    #####  angle transmission
    
    
    angleNode = root.createChild( "ANGLE" )
    
    r0 = 0.49
    r1 = 0.49
    
    body0 = StructuralAPI.RigidBody( angleNode, "body_0" )
    body0.setManually( [0,-4,0,0,0,0,1], 1, [1,1,1] )
    body0.dofs.showObject = True
    body0.dofs.showObjectScale = r0*1.1
    body0.dofs.velocity="0 0 0 1 0 0"
    body0.node.createObject('Sphere', radius=r0)
        
    body1 = StructuralAPI.RigidBody( angleNode, "body_1" )
    body1.setManually( [1,-4,0,0,0,0,1], 1, [1,1,1] )
    body1.dofs.showObject = True
    body1.dofs.showObjectScale = r1*1.1
    body1.node.createObject('Sphere', radius=r1)
    
   
    body0.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 0 1 1")
    body1.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 1 0 1")
    
    
    d = body0.node.createChild( "d" )
    d.createObject('MechanicalObject', template = 'Vec1'+StructuralAPI.template_suffix, name = 'dofs', position = '0 0 0' )
    input = [] # @internal
    input.append( '@' + Tools.node_path_rel(root,body0.node) + '/dofs' )
    input.append( '@' + Tools.node_path_rel(root,body1.node) + '/dofs' )
    d.createObject('GearMultiMapping', name = 'mapping', input = Tools.cat(input), output = '@dofs', pairs = "0 3 0 4", ratio = r0/r1 )
    body1.node.addChild( d )
    d.createObject('UniformCompliance', name = 'compliance', compliance="0" )
    
    
    
    
    #####  rack
    
    
    rackNode = root.createChild( "RACK" )
    
    
    body0 = StructuralAPI.RigidBody( rackNode, "body_0" )
    body0.setManually( [0,-6,0,0,0,0,1], 1, [1,1,1] )
    body0.dofs.showObject = True
    body0.dofs.showObjectScale = 0.55
    body0.dofs.velocity="0 0 0 0 0 1"
    body0.node.createObject('Sphere', radius=0.5)
        
    body1 = StructuralAPI.RigidBody( rackNode, "body_1" )
    body1.setManually( [-2,-6.71,0, 0,0,0.7071067811865476,0.7071067811865476], 1, [1,1,1] )
    body1.dofs.showObject = True
    body1.dofs.showObjectScale = 0.3
    body1.node.createObject('Capsule', radii="0.2", heights="5")
    
   
    body0.node.createObject('PartialFixedConstraint', fixedDirections="1 1 1 1 1 0")
    body1.node.createObject('PartialFixedConstraint', fixedDirections="0 1 1 1 1 1")
    
    
    d = body0.node.createChild( "d" )
    d.createObject('MechanicalObject', template = 'Vec1'+StructuralAPI.template_suffix, name = 'dofs', position = '0 0 0' )
    input = [] # @internal
    input.append( '@' + Tools.node_path_rel(root,body0.node) + '/dofs' )
    input.append( '@' + Tools.node_path_rel(root,body1.node) + '/dofs' )
    d.createObject('GearMultiMapping', name = 'mapping', input = Tools.cat(input), output = '@dofs', pairs = "0 5 0 0", ratio = 1 )
    body1.node.addChild( d )
    d.createObject('UniformCompliance', name = 'compliance', compliance="0" )
    
    