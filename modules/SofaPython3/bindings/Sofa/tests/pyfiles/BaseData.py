import unittest
import Sofa

class TestBaseData(unittest.TestCase):                    
        def test_ValidDataAccess(self):
                root = Sofa.Node("rootNode")
                c = root.createObject("MechanicalObject", name="t", position=[[0,0,0],[1,1,1],[2,2,2]])
                self.assertTrue(c.position is not None)
                
        def test_InvalidDataAccess(self):
                root = Sofa.Node("rootNode")
                self.assertRaises(AttributeError, getattr, root, "invalidData")

        def test_DataAsContainerAccess(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2]]
                c = root.createObject("MechanicalObject", name="t", position=v)                
                #self.assertEqual(list(c.position), v)
                self.fail("list(c.position) is an infinite loop.")
                self.assertEqual(len(c.position), 3)

        def test_DataAsContainerMemoryView(self):
                root = Sofa.Node("rootNode")
                v=[[0,0,0],[1,1,1],[2,2,2],[3,3,3]]
                c = root.createObject("MechanicalObject", name="t", position=v)                
                m = memoryview(c.position)           
                self.assertEqual(m.shape, (4,3))
                self.assertEqual(m[0,0], 0.0)
                self.assertEqual(m[1,1], 1.0)
                self.assertEqual(m[2,2], 2.0)
                self.assertEqual(m.tolist(), v)
                       
def createScene(rootNode):
        suite = unittest.TestLoader().loadTestsFromTestCase(TestBaseData)
        unittest.TextTestRunner(verbosity=2).run(suite)
