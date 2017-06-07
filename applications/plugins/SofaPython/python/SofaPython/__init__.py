import __builtin__
import sys

## @author Matthieu Nesme, Maxime Tournier
## @date 2017


# Keep a list of the modules always imported in the Sofa-PythonEnvironment
try:
    __SofaPythonEnvironment_importedModules__
except:
    __SofaPythonEnvironment_importedModules__ = sys.modules.copy()

    # some modules could be added here manually and can be modified procedurally
    # e.g. plugin's modules defined from c++
    __SofaPythonEnvironment_modulesExcludedFromReload = []



def unloadModules():
    """ call this function to unload python modules and to force their reload
        (useful to take into account their eventual modifications since
        their last import).
    """
    global __SofaPythonEnvironment_importedModules__
    toremove = [name for name in sys.modules if not name in __SofaPythonEnvironment_importedModules__ and not name in __SofaPythonEnvironment_modulesExcludedFromReload ]
    for name in toremove:
        del(sys.modules[name]) # unload it




import Sofa
class Controller(Sofa.PythonScriptController):

    def __init__(self, node, *args, **kwargs):
        Sofa.msg_warning('SofaPython', 'SofaPython.Controller is intended as compatibility class only')

        # setting attributes from kwargs
        for name, value in kwargs.iteritems():
            setattr(self, name, value)

        # call createGraph for compatibility purposes
        self.createGraph(node)


        # check whether derived class has 'onLoaded'
        cls = type(self)
        if not cls.onLoaded is Sofa.PythonScriptController.onLoaded:
            Sofa.msg_warning('SofaPython', 
                             '`onLoaded` is defined in subclass but will not be called in the future' )


        
