# -*- coding: utf-8 -*-

import Sofa
from SofaTest import *
def createScene(rootNode):
    print("Aliases: "+str(Sofa.getAliasesFor("VisualModel")))
    print("Aliases: "+str(Sofa.getAliasesFor("SphereModel")))
    print("Target for: "+str(Sofa.getAliasesFor("MechanicalObject")))
    ASSERT_EQ(type(Sofa.getTargetsFor("MechanicalObject")), list)
