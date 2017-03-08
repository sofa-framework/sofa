import __builtin__
import sys

class ImportFrame(object):
        """This class allow proper reloading of python modules.
           It work in the following way,

           At construction, this object saves the list of already loaded modules
           Then replace the import function with its own code.
           Then each time the import function is called the new code will record
           the new module loded.

           When the frame is uninstalled...all the module loaded between the construction
           and un-installation are remove from the list of loaded modules forcing a reload
           when needed further.
           
           WARNING, it is not possible to reload numpy this way... so we
           need to import it before.
        """
        def __init__(self):
                self.existingModules = sys.modules.copy()
                self.realImport = __builtin__.__import__
                __builtin__.__import__ = self.doImport
                self.moduleSet = {}

        def doImport(self, *args, **kwargs):
		mod = apply(self.realImport, args, kwargs)
		# This is a hack to exclude numpy from the reloading process.
		if args[0] == "numpy":
			if not args[0] in self.existingModules:
				self.existingModules[args[0]] = sys.modules[args[0]]

		self.moduleSet[args[0]] = (args, kwargs)
		return mod

        def uninstall(self):
                removed=[]
                for name in self.moduleSet:
                        if not self.existingModules.has_key(name) and name in sys.modules:
                                del(sys.modules[name])
                                removed.append(name)
                for name in removed:
                        del(self.moduleSet[name])


