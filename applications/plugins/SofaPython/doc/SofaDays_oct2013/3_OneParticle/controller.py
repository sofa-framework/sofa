import Sofa
import particle


class Tuto3(Sofa.PythonScriptController):

	# optionnally, script can create a graph...
	def createGraph(self,node):
		particle.oneParticleSample(node)
		return 0


