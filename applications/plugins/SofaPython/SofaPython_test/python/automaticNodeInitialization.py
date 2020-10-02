import Sofa
import SofaPython.Tools
import SofaTest


def createScene(node):
    node.createObject('PythonScriptController', filename=__file__, classname='VerifController')



class VerifController(SofaTest.Controller):

    def initGraph(self, node):

        Sofa.msg_info("initGraph ENTER")

        child = node.createChild("temporary_node")
        # FROM HERE, 'child' was added to the nodes to init in ScriptEnvironment, but it is not anymore

        node.removeChild( child )
        # 'child' is no longer in the scene graph but still was in ScriptEnvironment, but it is not anymore

        Sofa.msg_info("initGraph EXIT")

        # Coming back to SofaPython:
        # Nobody is no longer pointing to 'child', it will be deleted (smart pointer).
        # ScriptEnvironment was calling 'init' to an invalid pointer or
        # at least to a node detached from the scene graph,
        # but it does not anymore.
        # This could bring tons of potential troubles (including crashes).


    def onEndAnimationStep(self, dt):

        Sofa.msg_info("onEndAnimationStep")

        self.sendSuccess()
