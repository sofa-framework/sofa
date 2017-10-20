import sys
import Sofa
import Compliant


def createSceneAndController(root):

    global node
    node = root

    node.gravity="0 0 0"

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

    assembledSystem = Compliant.getImplicitAssembledSystem(node)
    print assembledSystem
    print "nb independent dofs:",assembledSystem.m
    print "nb constraints:",assembledSystem.n
    print "H", assembledSystem.getH()
    print "P", assembledSystem.getP()
    print "J", assembledSystem.getJ()
    print "C", assembledSystem.getC()

    sys.stdout.flush()

