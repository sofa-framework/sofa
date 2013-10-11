import Sofa

def vonMises3d(e):
	return max(0,0.5*( (e[0]-e[1])*(e[0]-e[1]) + (e[1]-e[2])*(e[1]-e[2]) + (e[2]-e[0])*(e[2]-e[0]) + 6.0*(e[3]*e[3]+e[4]*e[4]+e[5]*e[5]) ))**(0.5)
def norm(e):
	return (e[0]*e[0]+e[1]*e[1]+e[2]*e[2]+e[3]*e[3]+e[4]*e[4]+e[5]*e[5])**(0.5)

class ColorMap(Sofa.PythonScriptController):
	def createGraph(self,node):

		self.strain=node.getObject('E/E')			
		self.disp =node.getObject('disp')	

		return 0

	def onEndAnimationStep(self,dt):
		E = self.strain.findData('position').value
#		data = [ [vonMises3d(item)] for item in E]
		data = [ [norm(item)] for item in E]
		self.disp.findData('VoxelData').value= str(data)
		return 0

