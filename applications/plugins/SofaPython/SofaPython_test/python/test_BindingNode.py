# -*- coding: utf-8 -*-

import Sofa
import sys
from SofaTest import *

def createScene(rootNode):
    child1 = rootNode.createChild("child1")
    child2 = child1.createChild("child2")

    ## Check the 'generalized' getattr on Node
    child2.createObject("MechanicalObject", name="dofs", position=[[1,2,3], [4,5,6]], template="Vec3f")
    ASSERT_EQ( rootNode.child1.child2.dofs.position, [[1,2,3], [4,5,6]])

    ## Check the 'generalized' setattr on Node
    rootNode.child1.child2.dt = 1.0
    ASSERT_EQ( rootNode.child1.child2.dt, 1.0)

    ## Check that invalid path queries are generating exception.
    try:
        rootNode.notExists
    except Exception,e:
        pass
    else:
        raise Exception("Query for a non existing entry didn't returns an exception")


    ## Check that invalid path queries are generating exception.
    try:
        rootNode.child1.child2.dofs.positionq
    except Exception,e:
        pass
    else:
        raise Exception("Query for a non existing entry didn't returns an exception")

    ### A query for a non existing dofs should returns in a none object.
    ASSERT_EQ(child2.getObject("NOdofs", warning=False), None)
