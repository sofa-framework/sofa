import Sofa

# TODO standardize plugins python directory, then use something like
# Sofa.plugin_path('Compliant')

import sys
sys.path.append( Sofa.src_dir() + '/applications/plugins/Compliant/python' )

from Compliant import Rigid

def createScene(root):
    
    # root node setup
    root.createObject('RequiredPlugin', name = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    
    # simuation parameters
    root.dt = 1e-2
    root.gravity = [0, -9.8, 0]
    
    # ode solver
    ode = root.createObject('AssembledSolver')
    ode.stabilization = True
    
    # numerical solver
    num = root.createObject('MinresSolver')
    num.iterations = 500
    
    # scene node
    scene = root.createChild('scene')
    
    # script variables
    n = 10
    length = 2
    
    # objects/nodes creation
    obj = []
    node = []
    
    for i in xrange(n):
	# rigid bodies
        o = Rigid.Body()
        o.name = 'link-' + str(i)
        o.dofs.translation = [0, length * i, 0]
        obj.append( o )
        # insert the object into the scene node, saves the created
        # node in body_node
        body_node = o.insert(scene)
        node.append( body_node )

    # joints creation
    for i in xrange(n-1):
	# the joint
        j = Rigid.SphericalJoint()
        
        # joint offset definitions
        up = Rigid.Frame()
        up.translation = [0, length /2 , 0]
        
        down = Rigid.Frame()
        down.translation = [0, -length /2 , 0]
        
        # append node/offset to the joint
        j.append( node[i], up ) # parent
        j.append( node[i+1], down) # child
        
        j.insert(scene)
    
    # attach first node
    node[0].createObject('FixedConstraint', indices='0')
    
    # and wheee !
