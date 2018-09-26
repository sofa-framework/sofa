import Sofa
import unittest
import sys

class TestNode(unittest.TestCase):                
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
                self.assertTrue(hasattr(root,"child1"))
                root.removeChild(c)
                self.assertFalse(hasattr(root,"child1"))
               
        def test_createObjectWithParam(self):
                root = Sofa.Node("rootNode")
                root.createObject("MechanicalObject", name="mechanical", position=[[0,0,0],[1,1,1],[2,2,2]])
                        
        def test_children_property(self):                
                root = Sofa.Node("rootNode")                               
                c = root.addChild(Sofa.Node("child1"))
                c = root.addChild(Sofa.Node("child2"))                
                self.assertEqual(len(root.children), 2) 

        def test_parents_property(self):                
                root = Sofa.Node("rootNode")                               
                c1 = root.addChild(Sofa.Node("child1"))
                c2 = root.addChild(Sofa.Node("child2"))
                d = c1.addChild(Sofa.Node("subchild"))
                d = c2.addChild(Sofa.Node("subchild"))
                self.assertEqual(len(d.parents), 2) 
                
        def test_data_property(self):
                root = Sofa.Node("rootNode")
                self.assertTrue(hasattr(root, "__data__"))
                self.assertGreater(len(root.__data__), 0)
                self.assertTrue("name" in root.__data__)
                self.assertFalse(hasattr(root.__data__, "invalidEntry"))
                self.assertTrue(isinstance(root.__data__, Sofa.DataDict))
                        
## If we run a test scene from sofa we can access to the created scene. 
def createScene(rootNode):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestNode)
        unittest.TextTestRunner(verbosity=2).run(suite)
