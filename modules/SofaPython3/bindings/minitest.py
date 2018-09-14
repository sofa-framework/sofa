#!/usr/bin/python3

import sys
import os
os.environ["SOFA_ROOT"] = "/home/bruno/dev/refactorPython3/build/lib"
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

## Register all the common component in the factory. 
SofaRuntime.importPlugin("SofaAllCommonComponents")

## Init the simulation singleton. 
SofaRuntime.reinit()
r = Sofa.test()

b = r[0]
#b.name = "damien"
print("B"+str(b))
print("B"+b.name)

b = r[1]
b.name = "damien"
print("N"+str(b))
print("N"+b.name)


for t in r:
        print("type: "+str(type(t)))
        print("  .getName: "+str(t.name))
        print("  .getData: "+str(t.getData("name")))
        d = t.getData("name")
        print("     value(getData): "+str(d))
        d = t.name
        print("     value(__getattr__): "+str(d))
        #print("       mapping protocol [0]: "+str(t.name[0]))
        #print("       mapping protocol [:]: "+str(t.name[:]))





        #t.name = "NewName"
        #print("     value(getData): "+str(d))

