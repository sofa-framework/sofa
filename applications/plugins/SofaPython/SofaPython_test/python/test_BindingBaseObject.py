# -*- coding: utf-8 -*-

import Sofa
from SofaTest import *

def createScene(rootNode):
    rootNode.createObject("MechanicalObject", name="dofs")

    ASSERT_EQ(type(rootNode.dofs.getCategories()), list)

    ASSERT_EQ(type(rootNode.dofs.getTarget()), str)
