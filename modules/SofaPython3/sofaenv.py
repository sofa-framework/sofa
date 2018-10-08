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

## Register all the common component in the factory. 
#SofaRuntime.importPlugin("SofaAllCommonComponents")

## Init the simulation singleton. 

        
