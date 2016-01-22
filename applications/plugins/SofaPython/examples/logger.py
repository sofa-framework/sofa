import Sofa

def createScene(node):

    Sofa.msg_info( "emitter1", "my message info" )
    Sofa.msg_warning( "emitter2", "my message warning" )
    Sofa.msg_error( "emitter3", "my message error" )
    Sofa.msg_fatal( "emitter4", "my message fatal" )

    Sofa.msg_info( "my message info" )
    Sofa.msg_warning( "my message warning" )
    Sofa.msg_error( "my message error" )
    Sofa.msg_fatal( "my message fatal" )