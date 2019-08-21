import Sofa
import sys

def createScene(node):

    node.createObject('MechanicalObject', template="Vec3d", name="dofs1")
    node.createObject('MechanicalObject', template="Vec3d", name="dofs2")
    node.createObject('ConstantForceField', template="Vec3d", name="cff1")
    node.createObject('ConstantForceField', template="Vec3d", name="cff2")


    print '\n### SingleLink ###'
    print "findLink", node.findLink('mechanicalState')
    print "getValueString", node.findLink('mechanicalState').getValueString()
    print "attr", node.mechanicalState
    node.mechanicalState = '@dofs1'
    print "set attr", node.mechanicalState

    print '\n### MultiLink ###'
    print "findLink", node.findLink('forceField')
    print "getValueString", node.findLink('forceField').getValueString()
    print "attr", node.forceField



    sys.stdout.flush()