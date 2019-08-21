import Sofa

from Compliant import Rigid


def createScene(root):
    root.createObject('RequiredPlugin', pluginName = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    
    root.dt = 0.001
    root.gravity = [0, -9.8, 0]
    
    ode = root.createObject('CompliantImplicitSolver')
    ode.stabilization = "pre-stabilization"
    
    num = root.createObject('MinresSolver')
    num.iterations = 500
    
    scene = root.createChild('scene')
    
    base = Rigid.Body('base')
    moving = Rigid.Body('moving')

    moving.inertia_forces = True

    moving.dofs.translation = [0, 2, 0]

    base_node = base.insert( scene );
    base_node.createObject('FixedConstraint', indices = '0')

    moving_node = moving.insert( scene );

    base_offset = Rigid.Frame()
    base_offset.translation = [0, 1, 0]
    
    moving_offset = Rigid.Frame()
    moving_offset.translation = [0, -1, 0]
    
    joint = Rigid.SphericalJoint()
    
    # only rotation dofs
    joint.append(base_node, base_offset)
    joint.append(moving_node, moving_offset)

    node = joint.insert(scene)
    
    node.createObject('UniformCompliance',
                      template = 'Vec6d',
                      compliance = 1e-3 )
      
    
