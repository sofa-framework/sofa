import sys
import Sofa
import Compliant


def createSceneAndController(root):

    global node
    node = root

    node.gravity="0 0 0"

    global simuNode
    simuNode = node

    node.createObject('CompliantImplicitSolver')
    node.createObject('LDLTSolver',schur=False)


    node.createObject('MechanicalObject',position="0 0 0  1 0 0  1 1 0" , showObject=True, showObjectScale=10)
    node.createObject('FixedConstraint',indices="0")
    node.createObject('UniformMass',totalMass=100 )
    node.createObject('StiffSpringForceField',spring="0 1 100 5 1.5   0 2 100 5 1.5   1 2 100 5 1.5")


    cnode = node.createChild("constraint")
    cnode.createObject('MechanicalObject')
    cnode.createObject('DifferenceMapping',pairs="0 1")
    cnode.createObject('UniformCompliance',compliance="1e-5")





def onEndAnimationStep(dt):

    global node

    toto = Compliant.getImplicitAssembledSystem(node)
    print toto
    print "nb independent dofs:",toto.m
    print "nb constraints:",toto.n
    print "H", toto.getH()
    print "P", toto.getP()
    print "J", toto.getJ()
    print "C", toto.getC()

    sys.stdout.flush()

