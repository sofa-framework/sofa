
from Compliant import types
import numpy as np

def createScene(node):

    obj = node.createObject('MechanicalObject',
                            template = 'Rigid',
                            position = '0 0 0 0 0 0 1',
                            name = 'dofs')

    # a rigid view on position vector
    pos = np.array(obj.position).view(dtype = types.Rigid3)

    print type(pos[0])

    # now we have all kinds of fancy accessors
    pos[0].center = np.ones(3)

    print pos
