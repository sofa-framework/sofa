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

    # uncomment to get the location where the deprecated class is created
    # import traceback; traceback.print_stack()

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
        Controller.kwargs.append( kwarg )

        node.createObject('PythonScriptController',
                          filename = filename,
                          classname = cls.__name__,
                          name = name)
        # note the previous calls callbacks onLoaded and createGraph

        try:
            return Controller.instances.pop() # let's trust the garbage collector
        except AttributeError:
            # if this fails, you need to call
            # Controller.onLoaded(self, node) in derived classes
            print "[SofaPython.script.Controller.__new__] instance not found, did you call 'SofaPython.script.Controller.onLoaded' on your overloaded 'onLoaded' in {} ?".format(cls)
            raise

    def onLoaded(self, node):
        Controller.instances.append(self)
        self.additionalArguments(Controller.kwargs.pop()) # let's trust the garbage collector

    def additionalArguments(self,kwarg):
        """ to handle optional constructor arguments before createGraph
        """
        pass
