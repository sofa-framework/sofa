import Sofa

def createScene(node):

    # simply build a python controller here
    node.createObject('PythonScriptController',
                      filename = __file__,
                      classname = 'Controller',
                      name = 'script' )
    
    return node


 
class Controller(Sofa.PythonScriptController):

    def onLoaded(self, node):
        self.node = node
        return 0
        
    def onBeginAnimationStep(self, dt):
        # send script event 'success' or 'failure' to return script result
        if self.node.getTime() > 1:
            self.node.sendScriptEvent('success', 0)
        return 0

