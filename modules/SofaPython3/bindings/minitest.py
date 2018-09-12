import sys
sys.path.append("./Sofa/package")
sys.path.append("./SofaRuntime/package")

import Sofa
import SofaRuntime

SofaRuntime.reinit()

r = Sofa.test()

b = r[0]
b.name = "damien"
b.createObject("MechanicalObject", "test")

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

