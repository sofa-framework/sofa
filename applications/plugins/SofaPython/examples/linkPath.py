import Sofa
import sys

def createScene(node):

    childNode = node.createChild("childNode")
    print childNode.getLinkPath()
    print childNode.findData('name').getLinkPath()

    dofs = childNode.createObject('MechanicalObject',name="dofs")
    print dofs.getLinkPath()
    print dofs.findData('position').getLinkPath()

    sys.stdout.flush()