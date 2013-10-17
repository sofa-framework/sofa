import Sofa

# TODO handle this more cleanly, i.e. standardize plugins python
# directory, then use something like Sofa.add_plugin_path('Compliant')

import sys
sys.path.append( Sofa.src_dir() + '/applications/plugins/Compliant/python' )

from Compliant import Rigid


def createScene(root):
    root.createObject('RequiredPlugin', name = 'Compliant')
    root.createObject('VisualStyle', displayFlags="showBehavior" )
    
    root.dt = 0.01
    root.gravity = [0, -9.8, 0]
    
    ode = root.createObject('AssembledSolver')
    ode.stabilization = True
    
    num = root.createObject('MinresSolver')
    num.iterations = 100
    
    scene = root.createChild('scene')
    
    n = 10
    length = 2

    # objects/nodes creation
    obj = []
    node = []

    for i in xrange(n):
        o = Rigid.Body()
        o.name = 'link-' + str(i)
        o.dofs.translation = [0, length * i, 0]
        obj.append( o )
        # insert the object into the scene node, saves the created
        # node
        n = o.insert(scene)
        node.append( n )
    
    # joints creation
    for i in xrange(n-1):
        j = Rigid.SphericalJoint()

        # joint offsets
        up = Rigid.Frame()
        up.translation = [0, length /2 , 0]

        down = Rigid.Frame()
        down.translation = [0, -length /2 , 0]

        # node/offset
        j.append( node[i], up ) # parent
        j.append( node[i+1], down) # child
    
        j.insert(scene)
    
    # fix first node
    node[0].createObject('FixedConstraint', indices='0')
