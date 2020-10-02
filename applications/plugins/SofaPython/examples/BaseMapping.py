import Sofa
import sys
import numpy

def createSceneAndController(node):

    node0 = node.createChild('dof0')
    node0.createObject('MechanicalObject')

    node1 = node.createChild('dof1')
    node1.createObject('MechanicalObject')

    node01 = node0.createChild('dof01')
    node01.createObject('MechanicalObject')
    global mapping01
    mapping01 = node01.createObject('SubsetMultiMapping', input=node0.getLinkPath()+" "+node1.getLinkPath(), output="@.", indexPairs="0 0 1 0")
    node1.addChild(node01)

    node00 = node0.createChild('dof00')
    node00.createObject('MechanicalObject')
    global mapping00
    mapping00 = node01.createObject('IdentityMapping')


def bwdInitGraph(node):

    # simple mapping
    global mapping00
    print "simple mapping:", mapping00.getJs()
    print "as numpy:", numpy.asarray(mapping00.getJs()[0])


    # multimapping
    global mapping01
    Js01 = mapping01.getJs()
    print "multimapping:",Js01
    Js01_np = numpy.asarray(mapping01.getJs())
    print "as numpy:"
    print "J0:",Js01_np[0]
    print "J1:",Js01_np[1]



    sys.stdout.flush()


