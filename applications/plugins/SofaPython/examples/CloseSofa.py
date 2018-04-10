import Sofa
import sys

class CloseSofa(Sofa.PythonScriptController):

    def __init__(self, node):
        self.root = node
        
        self.createGraph(node)
        return None

    # optionnally, script can create a graph...
    def createGraph(self,node):
            node.getRootContext().animate = True

    def onEndAnimationStep(self, deltaTime):
        try :
            Sofa.unload(self.root)
            self.root.reset()
        except:
            print "something went off"
        else :
            print "All good"
        quit()
        return 0

def createScene(node):
    pyController = CloseSofa(node)
    return 0
