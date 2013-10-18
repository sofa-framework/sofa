import Sofa

# TODO handle this more cleanly, i.e. standardize plugins python
# directory, then use something like Sofa.add_plugin_path('Compliant')

import sys
sys.path.append( Sofa.src_dir() + '/applications/plugins/Compliant/python' )

from Compliant import Rigid


def createScene(root):
    root.createObject('RequiredPlugin', name = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    
    root.dt = 0.001
    root.gravity = [0, -9.8, 0]
    
    ode = root.createObject('AssembledSolver')
    ode.stabilization = True
    
    num = root.createObject('MinresSolver')
    num.iterations = 500
    
    scene = root.createChild('scene')
    
    base = Rigid.Body('base')
    moving = Rigid.Body('moving')

    moving.dofs.translation = [0, 2, 0]

    base_node = base.insert( scene );
    base_node.createObject('FixedConstraint', indices = '0')

    moving_node = moving.insert( scene );

    base_offset = Rigid.Frame()
    base_offset.translation = [0, 1, 0]
    
    moving_offset = Rigid.Frame()
    moving_offset.translation = [0, -1, 0]
    
    joint = Rigid.Joint()
    
    # only rotation dofs
    joint.append(base_node, base_offset)
    joint.append(moving_node, moving_offset)
    
    dofs = [0, 0, 0, 1, 1, 1]

    stiffness = 5e3
    value = Rigid.concat( [x / stiffness for x in dofs ] )
    
    add_compliance = False
    info = joint.insert(scene, add_compliance)
    
    # Node.removeChild does not work :-/
    
    compliance = info.node.createObject('DiagonalCompliance',
                                        name = 'compliance',
                                        template = 'Vec6d',
                                        compliance = value)
    
    # don't stabilize rotational compliance
    stab = info.node.createObject('Stabilization', mask = '1 1 1 0 0 0' )
    

    
  
    
