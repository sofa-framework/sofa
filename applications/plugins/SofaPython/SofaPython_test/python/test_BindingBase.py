# -*- coding: utf-8 -*-

import Sofa
from SofaTest import *

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

    ### addNewData
    lastCount = len(rootNode.getListOfDataFields())
    rootNode.addNewData("aData", "CustomData", "This is an help message", "f", 1.0)
    ASSERT_EQ(lastCount+1, len(rootNode.getListOfDataFields()))

    ASSERT_NEQ(rootNode.getData("aData"), None)
