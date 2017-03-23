'''simpler python script controllers'''

import Sofa
import inspect

class Controller(Sofa.PythonScriptController):

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