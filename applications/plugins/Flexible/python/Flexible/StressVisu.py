## @package Stress Visu
# Some PythonScriptControllers to visualize strain/ stress


import Sofa


# helper functions
def vonMises3d(e):
	return max(0,0.5*( (e[0]-e[1])*(e[0]-e[1]) + (e[1]-e[2])*(e[1]-e[2]) + (e[2]-e[0])*(e[2]-e[0]) + 6.0*(e[3]*e[3]+e[4]*e[4]+e[5]*e[5]) ))**(0.5)

def vonMises2d(e):
	return max(0,e[0]*e[0] + e[1]*e[1] - e[0]*e[1] + 3.0*e[2]*e[2])**(0.5)

def norm(e):
	return (e[0]*e[0]+e[1]*e[1]+e[2]*e[2]+e[3]*e[3]+e[4]*e[4]+e[5]*e[5])**(0.5)



## vizualisation of deformation field, discretized in an image
#  see example Flexible/examples/visualization/strainDiscretizer.scn

class ColorMap_Image(Sofa.PythonScriptController):

	def createGraph(self,node):

		self.strain=node.getObject('E/E')			
		self.disp =node.getObject('disp')	

		return 0

	def onEndAnimationStep(self,dt):
		E = self.strain.findData('position').value
#		data = [ [vonMises3d(item)] for item in E]
		data = [ [norm(item)] for item in E]
		self.disp.findData('VoxelData').value= str(data)
		self.disp.reinit()
		return 0



## vizualisation of frame deformations
#  see example:  Flexible/examples/visualization/linearAffineFrame_stressDisplay.scn

class ColorMap_Frame(Sofa.PythonScriptController):
	def createGraph(self,node):
		path = '@../'+self.variables[0][0]
		staticmesh = path+'/mesh'

		self.strain=node.getObject('E')			
		self.gpSampler=node.getParents()[0].getObject('sampler')			

		self.mynode = node.createChild('ColorMapping')
		self.mynode.createObject('TriangleSetTopologyContainer',name='mesh', src=staticmesh)
		self.indices=self.mynode.createObject('ImageValuesFromPositions',template='ImageUI',name='indices', position='@mesh.position', image='@../../sampler.region', transform='@../../sampler.transform' ,interpolation='0')
		self.disp =self.mynode.createObject('DataDisplay',maximalRange="false")
		self.mynode.createObject('ColorMap',colorScheme='Blue to Red', showLegend="1")
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



## vizualisation of quad mesh deformations
# see example: Flexible/examples/material/anisotropic2DFEM.scn

class ColorMap_Quad(Sofa.PythonScriptController):

	def createGraph(self,node):
		path = '@../../..'

		self.strain=node.getObject('E')			
		self.gpSampler=node.getParents()[0].getObject('sampler')			

		self.mynode = node.createChild('ColorMapping')
		self.mynode.createObject('TriangleSetTopologyContainer' )
  		self.mynode.createObject('TriangleSetTopologyModifier' )
  		self.mynode.createObject('Quad2TriangleTopologicalMapping' )
		self.disp =self.mynode.createObject('DataDisplay',maximalRange="false")
		self.mynode.createObject('ColorMap',colorScheme='Blue to Red', showLegend="1")
		self.mynode.createObject('IdentityMapping',input=path, output='@.')
		return 0

	def onEndAnimationStep(self,dt):
		S = self.strain.findData('force').value
		vol= self.gpSampler.findData('volume').value
		L = (len(S)/2)*[0]
		for index, item in enumerate(S):
			val = vonMises2d(item) / (vol[index][0] * 4.0)
        		L[2*(index/4)] += val
        		L[2*(index/4)+1] += val
		self.disp.findData('cellData').value= str(L).replace('[', '').replace("]", '').replace(",", ' ')
		return 0


## vizualisation of triangles mesh deformations
# see example: Flexible/examples/material/anisotropic2DFEM.scn

class ColorMap_Tri(Sofa.PythonScriptController):

	def createGraph(self,node):
		path = '@../../..'
		staticmesh = path+'/mesh'

		self.strain=node.getObject('E')			
		self.gpSampler=node.getParents()[0].getObject('sampler')			

		self.mynode = node.createChild('ColorMapping')
		self.mynode.createObject('TriangleSetTopologyContainer',name='mesh', src=staticmesh)
		self.disp =self.mynode.createObject('DataDisplay',maximalRange="false")
		self.mynode.createObject('ColorMap',colorScheme='Blue to Red', showLegend="1")
		self.mynode.createObject('IdentityMapping',input=path, output='@.')
		return 0

	def onEndAnimationStep(self,dt):
		S = self.strain.findData('force').value
		vol= self.gpSampler.findData('volume').value
		L = [vonMises2d(item) / vol[index][0] for index, item in enumerate(S)]
		self.disp.findData('cellData').value= str(L).replace('[', '').replace("]", '').replace(",", ' ')
		return 0



