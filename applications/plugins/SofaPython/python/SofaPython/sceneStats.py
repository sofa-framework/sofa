import Sofa
import sys
from pydoc import locate

def printSceneStats(node):
    v = SceneStatsVisitor()
    node.executeVisitor(v)

    print "====== SCENE STATS ======"
    print "nodes:",v.nodes
    if v.multinodes: print "     multinodes:",v.multinodes
    print "objects:",v.objects
    for c in v.objectClass:
        if v.objectClass[c]:
            print "     "+c+":",v.objectClass[c]

    print "========================="


    sys.stdout.flush()




class SceneStatsVisitor(object):

    def __init__(self):

        self.nodes = 0
        self.multinodes = 0

        self.objects = 0

        self.objectClass = {
            "BaseMechanicalState":0,
            "BaseMapping":0,
            "BaseLoader":0,
            "DataEngine":0,
            "Topology":0,
            "VisualModel":0,
            "PythonScriptController":0,
            "PythonScriptDataEngine":0,
            }

    def processNodeTopDown(self,node):

        self.nodes += 1
        if len(node.getParents())>1:
            self.multinodes += 1

        obj = node.getObjects()
        self.objects += len(obj)


        for o in obj:
            for c in self.objectClass:
                if issubclass(type(o), locate("Sofa."+c) ):
                    self.objectClass[c] += 1

        sys.stdout.flush()


        return True

    def processNodeBottomUp(self,node):
        pass

    def treeTraversal(self):
        return -1 # dag

