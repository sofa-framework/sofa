from __future__ import print_function
from SofaPython import sparse

import numpy as np

import Sofa
Sofa.loadPlugin('Flexible')

def createScene(node):

    template = 'Affine'
    dofs = node.createObject('MechanicalObject', template = template, name="dof", 
                             showObject="true", showObjectScale="0.7", size = 1)
    dofs.init()
    
    mass = node.createObject('AffineMass', template = template)
    mass.init()
    mass.bwdInit()

    ref = np.identity(12)
    
    with sparse.data_view(mass, 'massMatrix') as m:
        assert (m == ref).all()

        m[10, 10] = 14
        ref[10, 10] = 14

        # assert our in-place modifications are reflected
        with sparse.data_view(mass, 'massMatrix') as mm:
            assert (mm == ref).all()

        m[0, 10] = 14
        ref[0, 10] = 14

        # sparsity change
        with sparse.data_view(mass, 'massMatrix') as mm:
            assert not (mm == ref).all()

    # modification commit happens here
    with sparse.data_view(mass, 'massMatrix') as m:
        assert (m == ref).all()

    try:
        # this must throw ValueError since m is read-only outside context
        m[1,1] = 10
        assert False
    except ValueError:
        pass
        
