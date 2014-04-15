import Sofa

import Test

def createScene(node):

    # simply build a python controller here
    node.createObject('PythonScriptController',
                      filename = __file__,
                      classname = 'Controller',
                      name = 'script' )
    
    return node


 
class Controller(Test.Controller):

    def onLoaded(self, node):
        self.node = node
        return 0
        
    def onBeginAnimationStep(self, dt):
        condition = True
        self.should(condition, 'this test should never fail')
        return 0

