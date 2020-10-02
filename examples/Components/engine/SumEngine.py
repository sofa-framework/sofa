import Sofa
import sys

def createScene(node):

    node.createObject('MechanicalObject',name="mo",position="0 0 0   1 0 0   2 0 0",externalForce="1 0 0  0 1 0  -3 2 -10")
    sumEngine = node.createObject('SumEngine', input="@mo.externalForce")
    sumEngine.init()
    print "net external forces =",sumEngine.output
    sys.stdout.flush()