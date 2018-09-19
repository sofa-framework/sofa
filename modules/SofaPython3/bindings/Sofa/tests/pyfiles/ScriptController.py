import unittest
import gc
import Sofa
import SofaRuntime

class MyController(Sofa.PythonController):
        """This is my custom controller
           when init is called from Sofa this should call the python init function
        """        
        inited = 0
        reinited = 0
        
        def init(self):
                print("Python::init()")
                #print(" => "+str(dir(self)))
                #self.inited += 1
                
        def reinit(self):
                print("Python::reinit()")
                #self.reinited += 1
  
# class TestScriptController(unittest.TestCase):
#         def test_constructor(self):
#                 c = Sofa.PythonController()
#                 c.init()
#                 c.reinit()

#         def test_inheritance(self):
#                 c = MyController()
#                 c.name = "MyController"
#                 c.init()
#                 c.reinit()
#                 self.assertTrue( hasattr(c, "inited") ) 
#                 self.assertTrue( hasattr(c, "reinited") ) 

#                 print(str(c.inited))
#                 print(str(c.reinited))
                
#                 self.assertEqual( c.inited, 1 ) 
#                 self.assertEqual( c.reinited, 1 ) 
#                 return c
                
#         def test_inheritance2(self):
#                 node = Sofa.Node.createNode("root")
#                 node.addObject( self.test_inheritance() ) 
#                 gc.collect()
                
#                 ## We init the node (and thus all its child)
#                 node.init() 
                
#                 o = r.getObject("MyController")
#                 self.assertTrue( hasattr(o, "inited") ) 
#                 self.assertTrue( hasattr(o, "reinited") ) 
#                 self.assertEqual( o.inited, 2 ) 
#                 self.assertEqual( o.reinited, 2 ) 

def createScene(rootNode):
        print(str(dir(Sofa)))
        c = MyController()
        c1 = rootNode.addPythonObject(c)        
        print("============= c ===================")
        c.init()
        print("============= c1 ===================")
        c1.init()
        print("============= root::init ===================")
        SofaRuntime.getSimulation().init(rootNode)
        print("============= root::init2 ===================")
        SofaRuntime.getSimulation().init(rootNode)
        print("============= done ===================")

        #SofaRuntime.
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestScriptController)
    # return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
    

