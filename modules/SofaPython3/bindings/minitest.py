#!/usr/bin/python3

import sys
import os
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")
sys.path.append("./SofaTypes/package")

import Sofa
import SofaRuntime
import SofaTypes

d = SofaTypes.Vec3d(1,2,3)
print(d)
print("COUCOU")
print(dir(d))
print(d.y)
print(d.xyz)
print("Done")
## Register all the common component in the factory. 
#SofaRuntime.importPlugin("SofaAllCommonComponents")

## Init the simulation singleton. 

        
