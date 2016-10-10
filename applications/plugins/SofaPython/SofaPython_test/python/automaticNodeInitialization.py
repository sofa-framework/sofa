import Sofa
import SofaPython.Tools
import SofaTest


def createScene(node):
    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')



class VerifController(SofaTest.Controller):

    def initGraph(self, node):

        Sofa.msg_info("initGraph ENTER")

        child = node.createChild("temporary_node")
        # FROM HERE, 'child' is added to the node to init in ScriptEnvironment

        node.removeChild( child )
        # 'child' is no longer in the scene graph but it is still in ScriptEnvironment

        Sofa.msg_info("initGraph EXIT")

        # Coming back to SofaPython:
        # Nobody is no longer pointing to 'child', it will be deleted (smart pointer).
        # ScriptEnvironment try to call 'init' to an invalid pointer or
        # at least to a node detached from the scene graph.
        # This can bring tons of potential troubles.
        # ==> CRASH


    def onEndAnimationStep(self, dt):

        Sofa.msg_info("onEndAnimationStep")

        self.sendSuccess()
