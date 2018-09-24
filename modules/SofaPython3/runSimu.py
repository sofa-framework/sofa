#!/usr/bin/python3
import sys
import os

if "SOFA_ROOT" not in os.environ:
        print("WARNING: missing SOFA_ROOT in you environment variable. ") 
        sys.exit(-1)

sys.path.append(os.path.abspath("./bindings/Sofa/package"))
sys.path.append(os.path.abspath("./bindings/SofaRuntime/package"))
sys.path.append(os.path.abspath("./bindings/SofaTypes/package"))

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

        
        
        
