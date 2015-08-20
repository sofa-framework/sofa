
from SofaPython import script
from Compliant import future
from SofaPtyhon import Quaternion

import numpy as np
import math

def createScene( node ):

    # old school 
    obj = node.createObject('MechanicalObject',
                            template = 'Rigid3f',
                            name = 'dofs',
                            position = '0 0 0 0 0 0 1' )

    # obj is a future.Rigid3Object 
    print type(obj)
    
    pos = obj.position

    # pos[0] is a future.Rigid3 
    print type(pos[0])

    # we can do all kinds of fancy stuff with it
    x = math.pi / 3.0 * np.array([0, 1, 0])
    pos[0].orient = SofaPtyhon.Quaternion.exp(x)

    # still need this to commit changes, as everything is done through
    # Data get/set value
    obj.position = pos
    print obj.position

    # new school 
    obj2 = future.Rigid3Object(node, position = obj.position)

