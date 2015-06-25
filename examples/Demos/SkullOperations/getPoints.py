import Sofa

class CollisionPoints(Sofa.PythonScriptController):
	# key and mouse events; use this to add some user interaction to your scripts 
	def onKeyPressed(self,k):
		print 'onKeyPressed '+ord(k)
		return 0 
		
	def initGraph(self, node):
		self.here = node;
		self.object2track = node.getObject('DOFs');
		self.omni = node.getObject('omniDriver1');
		self.firstClic1 = 1;
		self.firstClic2 = 1;
		#Créqation de la sphère
		self.sphere = self.here.createChild('Sphere');
		self.visualsphere = self.sphere.createObject('OglModel', template='ExtVec3f', name='SVM',fileMesh='C:/Users/SOFA_GPU/Desktop/SkullOperations/meshs/PieceA.obj', scale3d='0.1 0.1 0.1');
		self.mechObj = self.sphere.createObject('MechanicalObject', template='Rigid3d', name='SMO', mapConstraints=1, mapForces=1, mapMasses=1);
		self.mapping = self.sphere.createObject('RigidMapping', template='Rigid3d,ExtVec3f', input='@./SMO', output='@./SVM');
		
		#Récupération de l'image MC
		self.container = node.getObject('viewer'); 
		print self.container.name;
		
		print 'OK initGraph';
		return 0		
			
	def onBeginAnimationStep(self,dt):
		if(self.omni.findData('stateButton1').value) and self.firstClic1:
			print 'Bouton sombre presse'
			self.firstClic1 = 0
		elif(not self.omni.findData('stateButton1').value) and not self.firstClic1:
			self.firstClic1 = 1
		if(self.omni.findData('stateButton2').value):
			if(self.firstClic2):
				print 'Bouton clair presse'
				self.firstClic2 = 0
			self.mechObj.position = self.object2track.position[0]
		elif(not self.omni.findData('stateButton2').value) and not self.firstClic2:
			self.firstClic2 = 1;
			#self.sphere.removeObject(self.visualsphere)
			#self.here.removeChild(self.sphere)
		return 0