import Sofa

def createScene(rootNode):
    Sofa.msg_info("Message with emitter='SofaPython'")
    Sofa.msg_info("Hello", "Message with emitter='Hello'")
    Sofa.msg_info(rootNode, "Message with emitter='rootNode'")
    Sofa.msg_error("Message with emitter='SofaPython'")
    Sofa.msg_error("Hello", "Message with emitter='Hello'")
    Sofa.msg_error(rootNode, "Message with emitter='rootNode'")
