import Sofa

# This example should be started from current dir, otherwise you need
# to adapt the following path to Compliant/python

# TODO fix this :)

import sys
sys.path.append('../python')

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
        node.append( o.insert(scene) )
    
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
