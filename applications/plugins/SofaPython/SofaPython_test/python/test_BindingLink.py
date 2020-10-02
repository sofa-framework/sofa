# -*- coding: utf-8 -*-

import Sofa
import SofaTest

def createScene(rootNode):
    rootNode.createChild("CHILD1")
    ASSERT_GT(len(rootNode.getListOfLinks()), 0)

    aField = rootNode.getListOfLinks()[0]
    ASSERT_EQ( aField.value, "@/CHILD1")

    aField.setPersistant(False)
    ASSERT_FALSE(aField.isPersistant())

    aField.setPersistant(True)
    ASSERT_TRUE(aField.isPersistant())
