import Sofa
import SofaPython.SofaNumpy
import sys


def createScene(node):

    mo = node.createObject("MechanicalObject",position="1 2 3 4 5 6")

    # regular Data accessor performs a copy in a list
    pos_copy_in_list = mo.position
    print type(pos_copy_in_list)

    # since it is a copy, it is not possible to directly change a value :(
    print( "before:",mo.position[0][0] )
    mo.position[0][0] = 123
    print( "after:",mo.position[0][0] )


    # accessing the Data as a numpy array is sharing c++ memory
    pos = SofaPython.SofaNumpy.numpy_data( mo, "position" )
    print type(pos)


    # since memory is shared, it is possible to directly change a value :)
    print( "before:",mo.position[0][0] )
    pos[0][0] = 123
    print( "after:",mo.position[0][0] )


    sys.stdout.flush()