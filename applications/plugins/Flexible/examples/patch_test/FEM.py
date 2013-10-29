#!/usr/bin/python
import math
import Sofa

def tostr(L):
	return str(L).replace('[', '').replace("]", '').replace(",", ' ')

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
	rootNode.createObject('VisualStyle', displayFlags="showBehaviorModels")

	restpos = [[0, 0, 0],   [1, 0, 0],   [0, 1, 0],   [1, 1, 0],   [0, 0, 1],   [1, 0, 1],   [0, 1, 1],   [1, 1, 1]]
	pos = [transform(T,item) for item in restpos]

	###########################################################
	simNode = rootNode.createChild('Hexa_barycentric')
	simNode.createObject('MeshTopology', name="mesh", position=tostr(restpos), hexahedra="0 1 3 2 4 5 7 6")
	simNode.createObject('MechanicalObject', template="Vec3d", name="parent", rest_position="@mesh.position",position=tostr(pos) )

	simNode.createObject('BarycentricShapeFunction', position="@parent.rest_position", nbRef="8")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('LinearMapping', template="Vec3d,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('LinearMapping', template="Vec3d,F331", showDeformationGradientScale="1")

 	childNode = simNode.createChild('Visu')
	childNode.createObject('VisualModel', color="8e-1 8e-1 1 1e-1")
	childNode.createObject('IdentityMapping')
 	childNode = simNode.createChild('Visu2')
	childNode.createObject('VisualStyle', displayFlags="showWireframe")
	childNode.createObject('VisualModel', color="8e-1 8e-1 1 1")
	childNode.createObject('IdentityMapping')

	simNode.createObject('PythonScriptController',filename="FEM.py", classname="Controller")

	###########################################################
	simNode = rootNode.createChild('Tetra_barycentric')
	simNode.createObject('MeshTopology', name="mesh", position=tostr(restpos), tetrahedra="0 5 1 7 0 1 2 7 1 2 7 3 7 2 0 6 7 6 0 5 6 5 4 0")

	simNode.createObject('MechanicalObject', template="Vec3d", name="parent", rest_position="@mesh.position",position=tostr(pos) )

	simNode.createObject('BarycentricShapeFunction', position="@parent.rest_position", nbRef="4")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('LinearMapping', template="Vec3d,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('LinearMapping', template="Vec3d,F331")

	simNode.createObject('PythonScriptController',filename="FEM.py", classname="Controller")


	###########################################################
	simNode = rootNode.createChild('Hexa_shepard')
	simNode.createObject('MeshTopology', name="mesh", position=tostr(restpos), hexahedra="0 1 3 2 4 5 7 6")
	simNode.createObject('MechanicalObject', template="Vec3d", name="parent", rest_position="@mesh.position",position=tostr(pos) )

	simNode.createObject('ShepardShapeFunction', position="@parent.rest_position", power="2")

	childNode = simNode.createChild('childP')
 	childNode.createObject('MechanicalObject',  template="Vec3d", name="child", position=tostr(samples) , showObject="1")
	childNode.createObject('LinearMapping', template="Vec3d,Vec3d")

	childNode = simNode.createChild('childF')
 	childNode.createObject('GaussPointContainer', position=tostr(samples))
 	childNode.createObject('MechanicalObject',  template="F331", name="child")
	childNode.createObject('LinearMapping', template="Vec3d,F331")

	simNode.createObject('PythonScriptController',filename="FEM.py", classname="Controller")

	###########################################################

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


