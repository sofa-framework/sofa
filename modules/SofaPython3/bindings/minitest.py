#!/usr/bin/python3

import sys
import os
os.environ["SOFA_ROOT"] = "/home/bruno/dev/refactorPython3/build/lib"
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

## Register all the common component in the factory. 
#SofaRuntime.importPlugin("SofaAllCommonComponents")

## Init the simulation singleton. 
SofaRuntime.reinit()

sys.path.append("./Sofa/tests/pyfiles")
import ScriptController

ScriptController.createScene(SofaRuntime.dev_getANode())
