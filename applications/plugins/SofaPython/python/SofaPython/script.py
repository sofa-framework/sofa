'''simpler python script controllers'''

import Sofa
import inspect

class Controller(Sofa.PythonScriptController):

    def __new__(cls, node, name='pythonScriptController'):

        node.createObject('PythonScriptController',
                          filename = inspect.getfile(cls),
                          classname = cls.__name__,
                          name = name)
        try:
            res = Controller.instance
            del Controller.instance
            return res
        except AttributeError:
            # if this fails, you need to call
            # Controller.onLoaded(self, node) in derived classes
            print "[SofaPython.script.Controller.__new__] instance not found, did you call 'SofaPython.script.Controller.onLoaded' on your overloaded 'onLoaded' in {} ?".format(cls)
            raise
        
    def onLoaded(self, node):
        Controller.instance = self
        
