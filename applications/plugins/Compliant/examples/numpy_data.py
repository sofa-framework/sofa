from __future__ import print_function

from Compliant import tool

def createScene(node):

    # some mechanical state
    state = node.createObject('MechanicalObject',
                              template = 'Rigid3d',
                              position = '0 0 0 1 0 0 0')

    # you need to load the plugin first 
    compliant = node.createObject('RequiredPlugin',
                                  pluginName = 'Compliant')

    # map position data directly as a numpy array
    m = tool.numpy_data(state, 'position')
    
    # before
    print(state.position)
    
    # modify data through a numpy array
    m[0][0] = 1

    # after
    print(state.position)

    # yay
