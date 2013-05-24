import Sofa

def vonMises3d(e):
	return max(0,0.5*( (e[0]-e[1])*(e[0]-e[1]) + (e[1]-e[2])*(e[1]-e[2]) + (e[2]-e[0])*(e[2]-e[0]) + 6.0*(e[3]*e[3]+e[4]*e[4]+e[5]*e[5]) ))**(0.5)

class ColorMap(Sofa.PythonScriptController):
	def createGraph(self,node):
		path = '@../'+self.findData('variables').value
		staticmesh = path+'/mesh'

		self.strain=node.getObject('E')			
		self.gpSampler=node.getParents()[0].getObject('sampler')			

		self.mynode = node.createChild('ColorMapping')
		self.mynode.createObject('TriangleSetTopologyContainer',name='mesh', src=staticmesh)
		self.indices=self.mynode.createObject('ImageValuesFromPositions',template='ImageUI',name='indices', position='@mesh.position', image='@../../sampler.region', transform='@../../sampler.transform' ,interpolation='0')
		self.disp =self.mynode.createObject('DataDisplay')
		self.mynode.createObject('ColorMap',colorScheme='Blue to Red')
		self.mynode.createObject('IdentityMapping',input=path, output='@.')
		return 0

	def onEndAnimationStep(self,dt):
		S = self.strain.findData('force').value
		vol= self.gpSampler.findData('volume').value
		numMoments= len(vol)/len(S)
		stress = [ vonMises3d(item) / vol[index*numMoments][0] for index, item in enumerate(S)]
		ind = self.indices.findData('values').value
		L = [ stress[int(item[0])-1] for item in ind]
		self.disp.findData('pointData').value= str(L).replace('[', '').replace("]", '').replace(",", ' ')
		return 0

