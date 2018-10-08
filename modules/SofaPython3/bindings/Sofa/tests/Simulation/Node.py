# -*- coding: utf-8 -*
import Sofa
import unittest
import sys

class Test(unittest.TestCase):
        def test_SimulationConstructor(self):
            root = Sofa.Simulation.Node("rootNode")
            self.assertEqual(root.name, "rootNode")

        def test_Constructor(self):
                root = Sofa.Node("rootNode")                               
                self.assertEqual(root.name, "rootNode")

        def test_GetAttr(self):
                root = Sofa.Node("rootNode")                               
                c = root.createChild("child1")
                self.assertTrue(c is not None)
                self.assertTrue(root.child1 is not None)
                
                o = c.createObject("MechanicalObject", name="mechanical")
                self.assertTrue(o is not None)
                self.assertTrue(root.child1.mechanical is not None)

        def test_init(self):
                root = Sofa.Node("rootNode")
                c = root.createChild("child1")
                c.createObject("MechanicalObject", position=[0.0,1.0,2.0]*100)
                root.init()

        def test_createChild(self):                
                root = Sofa.Node("rootNode")                               
                c = root.createChild("child1")
                self.assertTrue( c is not None )
                self.assertEqual( c.name, "child1" )
                self.assertTrue( hasattr(c, "child1") )

        def test_createChild(self):                
                root = Sofa.Node("rootNode")                               
                root.createChild("child1")
                self.assertTrue(hasattr(root,"child1"))
                
        def test_addChild(self):                
                root = Sofa.Node("rootNode")                               
                root.addChild(Sofa.Node("child1"))
                self.assertTrue(hasattr(root,"child1"))
                
        def test_removeChild(self):                
                root = Sofa.Node("rootNode")                               
                c = root.addChild(Sofa.Node("child1"))
                c2 = root.addChild(Sofa.Node("child2"))
                self.assertEqual(len(root.children), 2)
                self.assertTrue(hasattr(root,"child1"))
                self.assertTrue(hasattr(root,"child2"))
                root.removeChild(c)
                self.assertEqual(len(root.children), 1)
                self.assertFalse(hasattr(root,"child1"))
                root.removeChild("child2")
                self.assertFalse(hasattr(root,"child2"))
                self.assertEqual(len(root.children), 0)

        def test_createObjectWithParam(self):
                root = Sofa.Node("rootNode")
                root.createObject("MechanicalObject", name="mechanical", position=[[0,0,0],[1,1,1],[2,2,2]])
                        
        def test_children_property(self):                
                root = Sofa.Node("rootNode")                               
                c = root.addChild(Sofa.Node("child1"))
                self.assertEqual(len(root.children), 1)
                c = root.addChild(Sofa.Node("child2"))                
                self.assertEqual(len(root.children), 2) 

        def test_parents_property(self):                
                root = Sofa.Node("rootNode")                               
                c1 = root.addChild(Sofa.Node("child1"))
                c2 = root.addChild(Sofa.Node("child2"))
                d = c1.addChild(Sofa.Node("subchild"))
                d = c2.addChild(Sofa.Node("subchild"))
                self.assertEqual(len(d.parents), 1)
                c1.addChild(d)
                self.assertEqual(len(d.parents), 2)

        def test_objects_property(self):
                root = Sofa.Node("rootNode")
                root.createObject("MechanicalObject", name="name1")
                root.createObject("MechanicalObject", name="name2")
                self.assertEqual(len(root.objects), 2)
                root.createObject("MechanicalObject", name="name2")
                self.assertEqual(len(root.objects), 3)

        def test_data_property(self):
                root = Sofa.Node("rootNode")
                self.assertTrue(hasattr(root, "__data__"))
                self.assertGreater(len(root.__data__), 0)
                self.assertTrue("name" in root.__data__)
                self.assertFalse(hasattr(root.__data__, "invalidEntry"))
                self.assertTrue(isinstance(root.__data__, Sofa.Core.DataDict))

def getTestsName():
    suite = unittest.TestLoader().loadTestsFromTestCase(Test)
    return [ test.id().split(".")[2] for test in suite]

def runTests():
        import sys
        suite = None
        if( len(sys.argv) == 1 ):
            suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        else:
            suite = unittest.TestSuite()
            suite.addTest(Test(sys.argv[1]))
        return unittest.TextTestRunner(verbosity=1).run(suite).wasSuccessful()
