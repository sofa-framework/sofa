## @package IO
# Some functions to import/ export simulations
# see examples  Flexible/examples/IO/saveSimu.py  and    Flexible/examples/IO/loadSimu.py

import os

# helper functions
def datatostr(component,data):
	return str(component.findData(data).value).replace('[', '').replace("]", '').replace(",", ' ')

def affineDatatostr(data):
	L = ""
	for it in data :
		for i in xrange(3):
			L = L+ str(it[i])+" "
		L = L+ "["
		for i in xrange(9):
			L = L+ str(it[3+i])+" "
		L = L+ "] "
	return L




##   Write state of an 'affine' mechanicalObject in a python file 'filename'
# 'loadDofs(node)' function from this file allows to recreate the MechanicalObject
def export_AffineFrames(mechanicalObject, filename):
	f = open(filename, 'w')
	f.write("def loadDofs(node):\n\tcomponent=node.createObject('MechanicalObject', template='Affine',name='"+mechanicalObject.name+"'")
	f.write(", showObject='"+datatostr(mechanicalObject,'showObject')+"', showObjectScale='"+datatostr(mechanicalObject,'showObjectScale')+"'")
	f.write(", rest_position='"+affineDatatostr(mechanicalObject.findData('rest_position').value)+"'")
	f.write(", position='"+affineDatatostr(mechanicalObject.findData('position').value)+"'")
	f.write(")\n\treturn component\n")
	f.close()
	return 0

##   Write state of an 'affine' mechanicalObject in a python file 'filename'
# 'loadDofs(node)' function from this file allows to recreate the MechanicalObject
def export_RigidFrames(mechanicalObject, filename):
	f = open(filename, 'w')
	f.write("def loadDofs(node):\n\tcomponent=node.createObject('MechanicalObject', template='Rigid',name='"+mechanicalObject.name+"'")
	f.write(", showObject='"+datatostr(mechanicalObject,'showObject')+"', showObjectScale='"+datatostr(mechanicalObject,'showObjectScale')+"'")
	f.write(", rest_position='"+datatostr(mechanicalObject,'rest_position')+"'")
	f.write(", position='"+datatostr(mechanicalObject,'position')+"'")
	f.write(")\n\treturn component\n")
	f.close()
	return 0

##   Write state of a 'GaussPointSampler' in a python file 'filename'
# 'loadGPs(node)' function from this file allows to create a GaussPointContainer with similar points
def export_GaussPoints(gpsampler, filename):
	f = open(filename, 'w')
	volDim=1
	if isinstance(gpsampler.findData('volume').value, list) is True: # when volume is a list (several GPs or order> 1)
		volDim = len(gpsampler.findData('volume').value)/ len(gpsampler.findData('position').value)
	f.write("\ndef loadGPs(node):\n\tcomponent=node.createObject('GaussPointContainer',name='GPContainer'")
	f.write(", volumeDim='"+str(volDim)+"'")
	f.write(", inputVolume='"+datatostr(gpsampler,'volume')+"'")
	f.write(", position='"+datatostr(gpsampler,'position')+"'")
	f.write(")\n\treturn component\n")
	f.close()
	return 0


##   Export an ImageShapeFunction into images SF_indices.mhd and SF_weights.mhd
#    this creates two 'ImageExporter' in a node where the shape function 'sf' is located
#    optionally, a python file 'filename' is created to load the images using the function 'loadSF(node)'
def export_ImageShapeFunction(node, sf, filename):

    folder = os.path.dirname(filename)+"/"
    name = os.path.splitext(os.path.basename(filename))[0] # filename without extension

    node.createObject('ImageExporter', template="ImageUI", image="@"+sf.name+".indices", transform="@"+sf.name+".transform", filename=folder+name+"_indices.mhd", exportAtEnd="1", printLog="1")
    node.createObject('ImageExporter', template="ImageD", image="@"+sf.name+".weights", transform="@"+sf.name+".transform", filename=folder+name+"_weights.mhd", exportAtEnd="1", printLog="1")

    if filename!=0:
        f = open(filename, 'w')
        f.write("import os\ncurrentdir=os.path.dirname(os.path.realpath(__file__))\n\n")
        f.write("\ndef loadSF(node,name='"+name+"'):\n")
        f.write("\tnode.createObject('ImageContainer',template='ImageUI',name=name+'_indices',filename=currentdir+'/"+name+"_indices.mhd',drawBB='0')\n")
        f.write("\tnode.createObject('ImageContainer',template='ImageD',name=name+'_weights',filename=currentdir+'/"+name+"_weights.mhd',drawBB='0')\n")
        f.write("\tcomponent=node.createObject('ImageShapeFunctionContainer',name=name,transform='@'+name+'_weights.transform', weights='@'+name+'_weights.image', indices='@'+name+'_indices.image')\n")
        f.write("\treturn component\n")
        f.close()
	return 0


