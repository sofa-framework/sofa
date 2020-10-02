import Sofa
import SofaPython.SofaNumpy
import sys


def createSceneAndController(node):

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



    fc = node.createObject("FixedConstraint", fixAll=False, indices="0")
    print "a simple bool:", fc.fixAll, SofaPython.SofaNumpy.numpy_data( fc, "fixAll" )
    print "a simple scalar:", fc.drawSize, SofaPython.SofaNumpy.numpy_data( fc, "drawSize" )
    print "an array:", node.gravity, SofaPython.SofaNumpy.numpy_data( node, "gravity" )
    print "a 1D array:", fc.indices, SofaPython.SofaNumpy.numpy_data( fc, "indices" )
    print "a 2D array:", mo.position, SofaPython.SofaNumpy.numpy_data( mo, "position" )



    sys.stdout.flush()
