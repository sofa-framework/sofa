import __builtin__
import sys

## @author Matthieu Nesme
## @date 2017

def unloadModules():
    """ call this function to unload python modules and to force their reload
        (useful to take into account their eventual modifications since
        their last import).
    """
    global __SofaPython_moduleImport__
    __SofaPython_moduleImport__.unload()



################################################
### @internal the following should not be used by a regular user


class ModuleImport(object):
    """
       This class allows proper reloading of python modules.
       It works in the following way,

       At construction, this object saves the list of already loaded modules
       Then replaces the import function with its own code.
       Then each time the import function is called the new code will record
       the new loaded modules.

       When this instance is unloaded, all the modules loaded between the construction
       and un-installation are remove from the list of loaded modules forcing a reload
       when needed further.

       @author Damien Marchal
       @date 2017
    """


    def __init__(self):
        self.existingModules = sys.modules.copy() # the list of already imported modules that should never be unloaded
        self.realImport = __builtin__.__import__ # keep a hand on the real import function
        __builtin__.__import__ = self.doImport # switch import to our custom implementation
        self.moduleSet = [] # the list of newly imported modules

    def doImport(self, *args, **kwargs):
        # adding the newly imported module in the list
        self.moduleSet.append(args[0])
        # performing the real import
        return apply(self.realImport, args, kwargs)


    def unload(self):
        for name in self.moduleSet:
            if( not name in self.existingModules # ensure it was not an already imported modules
                    and name in sys.modules ): # ensure it is still imported
                del(sys.modules[name]) # unloaded it
        self.moduleSet = [] # fresh start


# creating a default global "ModuleImport" at the first import of this module
# performed only once in PythonEnvironment::init
try:
    __SofaPython_moduleImport__
except:
    __SofaPython_moduleImport__ = ModuleImport()

