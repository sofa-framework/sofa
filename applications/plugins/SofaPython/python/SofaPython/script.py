'''simpler & deprecated python script controllers'''

import Sofa
import inspect

def deprecated(cls):

    # TODO maybe we should print a backtrace to locate the origin?
    # or even better, use: https://docs.python.org/2/library/warnings.html#warnings.warn
    line = '''class `{0}` from module `{1}` is deprecated. You may now derive from `Sofa.PythonScriptController` and instantiate derived classes directly.'''.format(cls.__name__, cls.__module__)
    
    Sofa.msg_deprecated('SofaPython', line)
    
    Sofa.msg_deprecated('SofaPython', 
                        'note: `createGraph` will no longer be called automatically. You need to call manually from __init__ instead.')
    
    Sofa.msg_deprecated('SofaPython',
                        'note: `onLoaded` will no longer be called automatically. You need to call manually from __init__ instead.')
    
    return cls

@deprecated
class Controller(Sofa.PythonScriptController):
    
    # to stack data for recursive creations of Controllers
    instances = []
    kwargs = []

    def __new__(cls, node, name='pythonScriptController', filename='', **kwarg):
        """
        :param filename: you may have to define it (at least once) to create
                        a controller for which the class is defined in an external
                        file. Be aware the file will then be read several times.
        """

        # temporary variable to store optional arguments
        Controller.kwarg = kwarg

        node.createObject('PythonScriptController',
                          filename = filename,
                          classname = cls.__name__,
                          name = name)
        # note the previous calls callbacks onLoaded and createGraph

        # no need for storing optional arguments any longer
        del Controller.kwarg

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
        self.additionalArguments(Controller.kwarg)

    def additionalArguments(self,kwarg):
        """ to handle optional constructor arguments before createGraph
        """
        pass
