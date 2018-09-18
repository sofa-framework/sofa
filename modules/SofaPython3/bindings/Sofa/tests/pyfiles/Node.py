import Sofa
import unittest

class TestNode(unittest.TestCase):
        def test_base(self):
                r = Sofa.test()

                b = r[0]
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

