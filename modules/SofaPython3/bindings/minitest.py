import sys
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

SofaRuntime.reinit()
r = Sofa.test()

for t in r:
        print("type: "+str(type(t)))
        print("  .getName: "+str(t.getName()))
        print("  .getData: "+str(t.getData("name")))
        d = t.getData("name")
        print("     value: "+str(d))
