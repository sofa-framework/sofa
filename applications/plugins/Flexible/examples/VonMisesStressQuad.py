import Sofa

class Visualizer(Sofa.PythonScriptController):

	def createGraph(self,node):
		self.strain=node.getParents()[0].getObject('E')			
		self.gpSampler=node.getParents()[0].getParents()[0].getObject('sampler')			
		self.disp = node.createObject('DataDisplay')
		return 0

	def onEndAnimationStep(self,dt):
		S = self.strain.findData('force').value
		vol= self.gpSampler.findData('volume').value
		L = (len(S)/2)*[0]
		for index, item in enumerate(S):
			val = (max(0,item[0]*item[0] + item[1]*item[1] - item[0]*item[1] + 3.0*item[2]*item[2]))**(0.5) / (vol[index][0] * 4.0)
        		L[2*(index/4)] += val
        		L[2*(index/4)+1] += val
		self.disp.findData('cellData').value= str(L).replace('[', '').replace("]", '').replace(",", ' ')
		return 0

