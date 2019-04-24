# -*- coding: utf-8 -*-

import Sofa
from SofaTest import *


def test_addNewData(node):
    ### basic test of addNewData
    lastCount = len(node.getListOfDataFields())
    node.addNewData("aData", "CustomData", "This is an help message", "float", 1.0)
    ASSERT_EQ(lastCount+1, len(node.getListOfDataFields()))
    ASSERT_NEQ(node.getData("aData"), None)

    ### test we can create a rigid3 type.
    ASSERT_EQ(node.getData("aRigidData"), None)
    node.addNewData("aRigidData", "CustomData", "This is an help message", "Rigid3::VecCoord", [1.0,2.0,3.0,0.0,0.0,0.0,1.0])
    ASSERT_EQ(node.aRigidData, [[1.0,2.0,3.0,0.0,0.0,0.0,1.0]])

def createScene(rootNode):
    ## Check that the classical function are still there an returns not stupid results
    ASSERT_NEQ(0, rootNode.getDataFields())

    ### getListOfDataFields
    ASSERT_NEQ(0, rootNode.getListOfDataFields())
    for field in rootNode.getListOfDataFields():
        ASSERT_EQ(type(field), Sofa.Data)

    ### getListOfLinks
    ASSERT_NEQ(0, rootNode.getListOfLinks())
    for link in rootNode.getListOfLinks():
        ASSERT_EQ(type(link), Sofa.Link)

    ### getListOfLinks
    ASSERT_NEQ(0, rootNode.getListOfLinks())
    for link in rootNode.getListOfLinks():
        ASSERT_EQ(type(link), Sofa.Link)

    test_addNewData(rootNode)
