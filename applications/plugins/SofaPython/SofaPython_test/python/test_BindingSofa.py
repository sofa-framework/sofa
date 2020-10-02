# -*- coding: utf-8 -*-

import Sofa
from SofaTest import *
def createScene(rootNode):
    print("Aliases: "+str(Sofa.getAliasesFor("VisualModel")))
    print("Aliases: "+str(Sofa.getAliasesFor("SphereModel")))
    print("Components for target: "+str(Sofa.getComponentsFromTarget("SofaMiscCollision")))
    ASSERT_EQ(type(Sofa.getComponentsFromTarget("SofaMiscCollision")), list)
