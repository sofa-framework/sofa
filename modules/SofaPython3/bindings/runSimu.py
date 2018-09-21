#!/usr/bin/python3
import sys
import os
os.environ["SOFA_ROOT"] = "/home/bruno/dev/refactorPython3/build/lib"
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

## Register all the common component in the factory. 
#SofaRuntime.importPlugin("SofaPython3")
SofaRuntime.importPlugin("SofaAllCommonComponents")

if len(sys.argv) != 2:
        print("USAGE: python3 runSimu.py scene.py")
        sys.exit(-1)

## Init the simulation singleton. 
SofaRuntime.reinit()
c=SofaRuntime.load(sys.argv[1])
print("COUCOU2 "+c)
#SofaRuntime.getSimulation().init(rootNode)

        
