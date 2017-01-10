'''simpler python script controllers'''

import Sofa
import inspect

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