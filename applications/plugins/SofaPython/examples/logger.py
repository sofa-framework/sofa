import Sofa
import SofaPython.sml
import sys

def createScene(node):

    # some code before
    a = 0

# w/ emitter
    Sofa.msg_info( "MyPythonEmitter", "my message info  a="+str(a) )
    Sofa.msg_warning( "MyPythonEmitter  a="+str(a), "my message warning" )
    Sofa.msg_error( "MyPythonEmitter", "my message error" )
    Sofa.msg_fatal( "MyPythonEmitter", "my message fatal" )

    # some code in between
    a = 2

# w/o emitter
    Sofa.msg_info( "my message info  a="+str(a) )
    Sofa.msg_warning( "my message warning" )
    Sofa.msg_error( "my message error" )
    Sofa.msg_fatal( "my message fatal" )

    # more complex code was causing trouble, so try it
    model = SofaPython.sml.Model("smlSimple.xml")


    # # invalid calls
    # Sofa.msg_info( 100 )
    # Sofa.msg_info( "emitter", "message", "extra" )


    sys.stdout.flush()