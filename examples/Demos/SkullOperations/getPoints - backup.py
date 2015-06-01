import Sofa
from scipy import *
from numpy import *

class CollisionPoints(Sofa.PythonScriptController):
	# key and mouse events; use this to add some user interaction to your scripts 
	def onKeyPressed(self,k):
		print 'onKeyPressed '+ord(k)
		return 0 
		
	def initGraph(self, node):
		self.here = node;
		self.object2track = node.getObject('DOFs')
		self.omni = node.getObject('omniDriver1')
		self.firstClic1 = 1;
		self.firstClic2 = 1;
		self.i = 0;
		self.size = 0;
		self.listePoints = [0, 0, 0]
		print 'OK initGraph'
		return 0		
		
	#Crée une liste contenant les points d'un parallelepipede
	#Les attributs doivent etre des arrays
	def createPaveFromPoint(p1, p2, p3, p4):
		p12 = p2-p1;
		p13 = p3 - p1;
		n = cross(p12, p13)
		if(inner(n, array([0,-1,0]))<0):
			n = n * (-1);
		#Avoir un n pas trop gros
		norme = sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2])
		n=n/norme
		return 0
		
	def onBeginAnimationStep(self,dt):
		#print self.object2track.findData('position').value
		#print self.MSCO.findData('buttonDeviceState').value
		if(self.omni.findData('stateButton1').value) and self.firstClic1:
			print 'Bouton sombre presse'
			print self.listePoints
			self.firstClic1 = 0
		elif(not self.omni.findData('stateButton1').value) and not self.firstClic1:
			self.firstClic1 = 1
		if(self.omni.findData('stateButton2').value):
			if(self.firstClic2):
				print 'Bouton clair presse'
				self.firstClic2 = 0
				self.sphere = self.here.createChild('Sphere')
				self.visualsphere = self.sphere.createObject('OglModel', template='ExtVec3f', name='sphereVisualModel',fileMesh='mesh/sphere.obj', scale3d='0.1 0.1 0.1')
				if(self.size<3):
					self.listePoints[self.i] = self.object2track.position[0][0:3]
					self.i = self.i + 1
					self.size = self.size +1
			self.visualsphere.translation = self.object2track.position[0][0:3]
		elif(not self.omni.findData('stateButton2').value) and not self.firstClic2:
			self.firstClic2 = 1;
			#self.sphere.removeObject(self.visualsphere)
			self.here.removeChild(self.sphere)
		return 0