import unittest
import gc
import Sofa

class MyController(Sofa.PythonController):
        """This is my custom controller
           when init is called from Sofa this should call the python init function
        """        
        inited = 0
        reinited = 0
        
        def __init__(self):
                Sofa.PythonController.__init__(self)
                self.name = "Damien"
                #self.nulle = "Z"
                print(" Python::__init__::"+str(self.name))
        
        def __del__(self):
                print(" Python::__del__")
        
        def init(self):
                print(" Python::init() at "+str(self))
                #print(" => "+str(dir(self)))
                #self.inited += 1
                
        def reinit(self):
                print(" Python::reinit() at "+str(self))
                #self.reinited += 1
  
class Test(unittest.TestCase):
         def test_constructor(self):
                 c = Sofa.PythonController()
                 c.init()
                 c.reinit()

         def test_inheritance(self):
                 c = MyController()
                 c.name = "MyController"
                 c.init()
                 c.reinit()
                 self.assertTrue( hasattr(c, "inited") ) 
                 self.assertTrue( hasattr(c, "reinited") ) 

                 print(str(c.inited))
                 print(str(c.reinited))
                
                 self.assertEqual( c.inited, 1 ) 
                 self.assertEqual( c.reinited, 1 ) 
                 return c
                
         def test_inheritance2(self):
                 node = Sofa.Node.createNode("root")
                 node.addObject( self.test_inheritance() ) 
                 gc.collect()
                
                 ## We init the node (and thus all its child)
                 node.init() 
                
                 o = r.getObject("MyController")
                 self.assertTrue( hasattr(o, "inited") ) 
                 self.assertTrue( hasattr(o, "reinited") ) 
                 self.assertEqual( o.inited, 2 ) 
                 self.assertEqual( o.reinited, 2 ) 

def runTests():
        suite = unittest.TestLoader().loadTestsFromTestCase(Test)
        return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    
def createScene(rootNode):
        runTests()

