#!/usr/bin/python
import math
import Sofa

def tostr(L):
	return str(L).replace('[', '').replace("]", '').replace(",", ' ')
def concatAffineinString(p,A):
	ret=""
	for i,P in enumerate(p):
		ret+=tostr(P)+" ["+tostr(A[i])+"] "
	return ret


def transform(T,p):
	return [T[0][0]*p[0]+T[0][1]*p[1]+T[0][2]*p[2]+T[1][0],T[0][3]*p[0]+T[0][4]*p[1]+T[0][5]*p[2]+T[1][1],T[0][6]*p[0]+T[0][7]*p[1]+T[0][8]*p[2]+T[1][2]]
def transformF(T,F):
	return [T[0][0]*F[0]+T[0][1]*F[3]+T[0][2]*F[6],T[0][0]*F[1]+T[0][1]*F[4]+T[0][2]*F[7],T[0][0]*F[2]+T[0][1]*F[5]+T[0][2]*F[8],T[0][3]*F[0]+T[0][4]*F[3]+T[0][5]*F[6],T[0][3]*F[1]+T[0][4]*F[4]+T[0][5]*F[7],T[0][3]*F[2]+T[0][4]*F[5]+T[0][5]*F[8],T[0][6]*F[0]+T[0][7]*F[3]+T[0][8]*F[6],T[0][6]*F[1]+T[0][7]*F[4]+T[0][8]*F[7],T[0][6]*F[2]+T[0][7]*F[5]+T[0][8]*F[8]]

def compare(p1,p2):
	res = 0
	for i,P1 in enumerate(p1):
		for j,item in enumerate(P1):
			res = res+ (item-p2[i][j])*(item-p2[i][j])
	return res

ERRORTOL = 1e-5
T = [[2,0,0,0,2,0,0,0,2],[0,0,0]]
#T = [[0.8,1.2,0.3,0,1.9,0.45,0.5,2.8,0.2],[5,2,8]]
samples= [[0.5,0.5,0.5], [0.23,0.5,0.8], [0,0.12,0], [0.8,0,0.58]]

# scene creation method
def createScene(rootNode):
	rootNode.createObject('RequiredPlugin', pluginName="Flexible")
	rootNode.createObject('RequiredPlugin', pluginName="image")
	rootNode.createObject('VisualStyle', displayFlags="showBehaviorModels")

	restpos = [[0, 0, 0],   [1, 0, 0],   [0, 1, 0],   [1, 1, 0],   [0, 0, 1],   [1, 0, 1],   [0, 1, 1],   [1, 1, 1]]
	pos = [transform(T,item) for item in restpos]
	restAffinepos =  [[1,0,0,0,1,0,0,0,1] for item in restpos]
	Affinepos = [transformF(T,item) for item in restAffinepos]

	rootNode.createObject('MeshTopology', name="mesh", position="0 0 0   1 0 0   0 1 0   1 1 0   0 0 1   1 0 1   0 1 1   1 1 1", triangles="0 1 3    4 5 7   0 4 6   6 7 3   1 3 7   0 1 5 0 3 2    4 7 6   0 6 2   6 3 2   1 7 5   0 5 4")
	rootNode.createObject('MeshToImageEngine', template="ImageUC",name="rasterizer", src="@mesh", voxelSize="0.1", padSize="1", rotateImage="true")
#	rootNode.createObject('ImageViewer',src="@rasterizer")


	###########################################################
	simNode = rootNode.createChild('Point Shepard')

	simNode.createObject('MechanicalObject', template="Vec3d", name="parent", showObject="1", rest_position=tostr(restpos), position=tostr(pos))
	simNode.createObject('ShepardShapeFunction', name="SF", position="@parent.rest_position", power="2", nbRef="6")
	simNode.createObject('ShapeFunctionDiscretizer', name="SF3D", src="@../rasterizer")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('MLSMapping', template="Vec3d,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('MLSMapping', template="Vec3d,F331", showDeformationGradientScale="1")

	simNode.createObject('PythonScriptController',filename="MLS.py", classname="Controller")

	###########################################################
	simNode = rootNode.createChild('Point Hat')

	simNode.createObject('MechanicalObject', template="Vec3d", name="parent", showObject="1", rest_position=tostr(restpos), position=tostr(pos))
	simNode.createObject('HatShapeFunction', name="SF", position="@parent.rest_position", param="1.5	 2 3", nbRef="6")
	simNode.createObject('ShapeFunctionDiscretizer', name="SF3D", src="@../rasterizer")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('MLSMapping', template="Vec3d,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('MLSMapping', template="Vec3d,F331", showDeformationGradientScale="1")

	simNode.createObject('PythonScriptController',filename="MLS.py", classname="Controller")


	###########################################################
	simNode = rootNode.createChild('Affine Shepard')

	simNode.createObject('MechanicalObject', template="Affine", name="parent", showObject="1", rest_position=concatAffineinString(restpos,restAffinepos), position=concatAffineinString(pos,Affinepos))
	simNode.createObject('ShepardShapeFunction', name="SF", position="@parent.rest_position", power="2", nbRef="6")
	simNode.createObject('ShapeFunctionDiscretizer', name="SF3D", src="@../rasterizer")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('MLSMapping', template="Affine,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('MLSMapping', template="Affine,F331", showDeformationGradientScale="1")

	simNode.createObject('PythonScriptController',filename="MLS.py", classname="Controller")


	###########################################################
	simNode = rootNode.createChild('Affine Hat')

	simNode.createObject('MechanicalObject', template="Affine", name="parent", showObject="1", rest_position=concatAffineinString(restpos,restAffinepos), position=concatAffineinString(pos,Affinepos))
	simNode.createObject('HatShapeFunction', name="SF", position="@parent.rest_position", param="1.5 2 3", nbRef="6")
	simNode.createObject('ShapeFunctionDiscretizer', name="SF3D", src="@../rasterizer")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('MLSMapping', template="Affine,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('MLSMapping', template="Affine,F331", showDeformationGradientScale="1")

	simNode.createObject('PythonScriptController',filename="MLS.py", classname="Controller")

	rootNode.animate=1
	return rootNode


class Controller(Sofa.PythonScriptController):
	def createGraph(self,node):
		self.node=node
		self.done=0
		return 0

	def onEndAnimationStep(self,dt):

		if self.done==0:
			print "TEST "+self.node.name+":"
			# test points		
			restpos = self.node.getObject('childP/child').findData('rest_position').value
			refpos = [transform(T,item) for item in restpos]
			pos = self.node.getObject('childP/child').findData('position').value
			error = compare(refpos,pos)
			if error>ERRORTOL :
				print "\t"+"\033[91m"+"[FAILED]"+"\033[0m"+" error on P= "+str(error)
			else :
				print "\t"+"\033[92m"+"[OK]"+"\033[0m"+" error on P= "+str(error)

			# test defo gradients		
			restpos = [1,0,0,0,1,0,0,0,1]
			pos = self.node.getObject('childF/child').findData('position').value
			refpos = [transformF(T,restpos) for item in pos]
			error = compare(refpos,pos)
			if error>ERRORTOL :
				print "\t"+"\033[91m"+"[FAILED]"+"\033[0m"+" error on F= "+str(error)
			else :
				print "\t"+"\033[92m"+"[OK]"+"\033[0m"+" error on F= "+str(error)

			self.done=1

		return 0


