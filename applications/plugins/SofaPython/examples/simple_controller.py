'''example: a simpler interface to python controllers

this example shows how controllers can be defined and created very
easily by deriving script.Controller.

'''


import Sofa
from SofaPython import script


class EasyScript(script.Controller):

    def onBeginAnimationStep(self, dt):
        print 'easy', self.foo


class TediousScript(Sofa.PythonScriptController):

    def onBeginAnimationStep(self, dt):
        print 'tedious', self.foo, shared.bar

    def onLoaded(self, node):
        TediousScript.instance = self

# global shared variables
class Shared: pass

def createScene(node):

    # tedious
    obj = node.createObject('PythonScriptController',
                            filename = __file__,
                            classname = 'TediousScript' )

    # obj is an instance of Sofa.PythonScriptController, which does
    # not expose the TediousScript instance directly.

    # in order to communicate with the instance, one can just use
    # global variables, or use entry points to get the instance:

    global shared
    shared = Shared()
    shared.bar = 14
    
    tedious = TediousScript.instance
    tedious.foo = 42

    # script.Controller wraps it all to make it easier:
    easy = EasyScript(node)
    easy.foo = 42


    
