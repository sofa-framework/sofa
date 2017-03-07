import __builtin__
import sys 


class ImportFrame(object):
	def __init__(self):
		print("Installing an import frame")
		self.existingModules = sys.modules.copy()
		self.realImport = __builtin__.__import__
		__builtin__.__import__ = self.doImport
		self.moduleSet = {}

	def doImport(self, a,b,c,d):
		mod = apply(self.realImport, (a,b,c,d))
		self.moduleSet[a] = (a,b,c,d)
		return mod       

	def uninstall(self):
		print("UnInstalling an import frame")
		removed=[]
		for name in self.moduleSet:
			if not self.existingModules.has_key(name):
				del(sys.modules[name])
				removed.append(name)
		for name in removed:
			del(self.moduleSet[name])		

		
