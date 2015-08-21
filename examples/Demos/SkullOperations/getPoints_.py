#!/usr/bin/python2.6
import Sofa
#Aimed to be used for deleting cubes from the scene when i planned on making a surface of the skull with the sampler
class CollisionPoints(Sofa.PythonScriptController):
		
	def initGraph(self, node):
		self.here = node;
		self.object2track = node.getObject('./Omni/DOFs');
		self.omni = node.getObject('./Omni/omniDriver1');
		self.carving = node.getObject('./Crane/ctc');
		# self.firstClic1 = 1;
		# self.firstClic2 = 1;
		#Creation de la sphere
#		self.sphere = self.here.createChild('Sphere');
#		self.visualsphere = self.sphere.createObject('OglModel', template='ExtVec3f', name='SVM',fileMesh='C:/Users/SOFA_GPU/Desktop/SkullOperations/meshs/PieceA.obj', scale3d='0.1 0.1 0.1');
#		self.mechObj = self.sphere.createObject('MechanicalObject', template='Rigid3d', name='SMO', mapConstraints=1, mapForces=1, mapMasses=1);
#		self.mapping = self.sphere.createObject('RigidMapping', template='Rigid3d,ExtVec3f', input='@./SMO', output='@./SVM');
		
		print 'OK initGraph';
		return 0		
			
	# def bwdInitGraph(self,node):
		# print 'bwdInitGraph called (python side)'
		# Recuperation des elements des cubes
		# hexa = sampler.findData("hexahedra").value;
		# vertex = sampler.findData("position").value;
		
		# tmp = [0,1,2, 0,2,3, 0,1,5, 0,5,4, 1,2,6, 1,6,5, 3,2,6, 3,6,7, 0,3,7, 0,7,4, 7,4,5, 7,5,6];
		# cptFaces = {};
		# faces = {};
		# pour chaque cube
		# for i in hexa :
			# pour chaque triangle du cube
			# for j in range(0,11) :
				# je récupere les points des triangles
				# pos1= tmp[j*3+0];
				# pos2= tmp[j*3+1];
				# pos3= tmp[j*3+2];
				# Je les concatene
				# tmp = tuple(sorted([pos1, pos2, pos3]));
				# J'incremente la valeur dans la map
				# faces[tmp] = faces[tmp]+1;
				# J'ajoute la face dans un tableau
				# faces.append([])
		# pour toutes les faces dans le tableau
		# for f in 
		# return 0
			
			
	def onBeginAnimationStep(self,dt):
		# self.container = self.here.getObject('./cube/viewer'); 
		# print self.container.plane[1];
		# if(self.omni.findData('stateButton1').value) and self.firstClic1:
			# print 'Bouton sombre presse'
			# self.firstClic1 = 0
		# elif(not self.omni.findData('stateButton1').value) and not self.firstClic1:
			# self.firstClic1 = 1
		# if(self.omni.findData('stateButton2').value):
			# if(self.firstClic2):
				# print 'Bouton clair presse'
				# self.firstClic2 = 0
			# self.mechObj.position = self.object2track.position[0]
		# elif(not self.omni.findData('stateButton2').value) and not self.firstClic2:
			# self.firstClic2 = 1;
			#self.sphere.removeObject(self.visualsphere)
			#self.here.removeChild(self.sphere)
		return 0