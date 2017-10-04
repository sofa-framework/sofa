import sys
import numpy

import Sofa
import Compliant


def createSceneAndController(node):

    node.gravity="0 0 0"

    global simuNode
    simuNode = node

    node.createObject('CompliantImplicitSolver')
    node.createObject('LDLTSolver')


    node.createObject('MechanicalObject',position="0 0 0  1 0 0  1 1 0" , showObject=True)
    node.createObject('FixedConstraint',indices="0")
    node.createObject('UniformMass',totalMass=100 )
    node.createObject('StiffSpringForceField',spring="0 1 100 5 1.5   0 2 100 5 1.5   1 2 100 5 1.5")




def bwdInitGraph(node):

    # here M is constant
    M = Compliant.getAssembledImplicitMatrix(simuNode,1,0,0)
    print "M: ", numpy.asarray(M)

    sys.stdout.flush()


def onEndAnimationStep(dt):

    global simuNode

    # K changes at each step
    K = Compliant.getAssembledImplicitMatrix(simuNode,0,0,1)
    print "K: ", numpy.asarray(K)

    sys.stdout.flush()