import __builtin__
import sys

class ModuleImport(object):
    """This class allows proper reloading of python modules.
       It works in the following way,

       At construction, this object saves the list of already loaded modules
       Then replaces the import function with its own code.
       Then each time the import function is called the new code will record
       the new loaded modules.

       When this instance is unloaded, all the modules loaded between the construction
       and un-installation are remove from the list of loaded modules forcing a reload
       when needed further.

       WARNING, it is not possible to reload numpy this way... so we
       need to import it before.

       @author Damien Marchal
       @date 2017
    """


    # a list of modules that cannot be handle this way
    excludedModules = ["numpy"]



    def __init__(self):
        self.existingModules = sys.modules.copy()
        self.realImport = __builtin__.__import__
        __builtin__.__import__ = self.doImport
        self.moduleSet = {}

    def doImport(self, *args, **kwargs):
        mod = apply(self.realImport, args, kwargs)
        # This is a hack to exclude numpy from the reloading process.
        if args[0] in ModuleImport.excludedModules:
            if not args[0] in self.existingModules:
                self.existingModules[args[0]] = sys.modules[args[0]]

        self.moduleSet[args[0]] = (args, kwargs)
        return mod


    def unload(self):
        removed=[]
        for name in self.moduleSet:
            if not self.existingModules.has_key(name) and name in sys.modules:
                del(sys.modules[name])
                removed.append(name)
        for name in removed:
            del(self.moduleSet[name])
