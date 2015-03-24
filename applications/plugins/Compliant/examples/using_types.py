
from Compliant import types
import numpy as np

def createScene(node):

    obj = node.createObject('MechanicalObject',
                            template = 'Rigid',
                            position = '0 0 0 0 0 0 1',
                            name = 'dofs')

    pos = np.array(obj.position).view(dtype = types.Rigid3)

    print type(pos[0])

    pos[0].center = np.ones(3)

    print pos
