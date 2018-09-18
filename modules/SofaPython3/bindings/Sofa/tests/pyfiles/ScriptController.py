import unittest
import Sofa
import SofaRuntime

class MyController(Sofa.BaseObject):
        """This is my custom controller
           when init is called from Sofa this should call the python init function
        """
        
        def init(self):
                print ('MyController::init()')
                
        def reinit(self):
                print ('MyController::reinit()')
  
# class TestScriptController(unittest.TestCase):
#         def test_constructor(self):
#                 c = Sofa.PythonController()
#                 c.init()
#                 c.reinit()

#         def test_inheritance(self):
#                 c = MyController()
#                 c.init()
#                 rootNode.addObject(c)
#                 c.reinit()
#                 self.assertTrue( hasattr(c, "inited") ) 
#                 self.assertTrue( hasattr(c, "reinited") ) 
                
#         def test_inheritance(self):
#                 c = MyController()
#                 c.init()
#                 c.reinit()
#                 self.assertTrue( hasattr(c, "inited") ) 
#                 self.assertTrue( hasattr(c, "reinited") ) 

def createScene(rootNode):
        c = MyController()
        c = rootNode.addPythonObject(c)
        c.init()
        SofaRuntime.getSimulation().init(rootNode)

    # suite = unittest.TestLoader().loadTestsFromTestCase(TestScriptController)
    # return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()
