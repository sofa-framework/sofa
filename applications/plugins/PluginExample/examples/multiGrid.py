import Sofa
import math
import os

from decimal import Decimal


############################################################################################
# Recalage Multi grille
############################################################################################

#################################"  
# Fonctions utilisees plus tard :
#################################"
def isIn(n,sop):
	l=len(sop)
	i=0
	while (i<l):
		if (n==int(sop[i])):
			return True
		i+=1
	return False


class MultiGrid:
	
############################"  
# Definition des parametres :
############################"

	# Multigrille ?
	multi = True
	
	# Definition des ou de la Grille(s)
	if (multi):
		nGrid=4 # Nombre de grille pour le recalage (doit etre au moins egal a 2)
		nVal1=2 # Valeur des nx ny et nz de la plus grossiere des regularGrids
		if (nGrid<2):
			print "Erreur : le nombre de grilles doit etre au moins egal a 2"
	else:
		nGrid=1
		nVal1=8 # Valeur des nx ny et nz de la regularGrid
	nVal1Char=str(nVal1)
	
	# Nom du mesh utilise
	#nomMesh = 'Mesh/foie_grillefine'
	#nomMesh = 'test_cont_A'
	#nomMesh = 'test_cont_SGrA'
	nomMesh = 'Mesh/foie_GF_avec_point'
	#nomMesh = 'Mesh/foie_GF_avec_pointb'
	#nomMesh = 'Mesh/foie_GF_avec_pointc'
	#nomMesh = 'Mesh/foie_GF_avec_pointd'
	#nomMesh = 'Mesh/foie_GF_avec_pointe'
	#nomMesh = 'Mesh/foie'
	#nomMesh = 'Mesh/foieb'
	#nomMesh = 'Mesh/foiec'

	# Definition manuelle des ressorts ?
	manual = False
	
	# Definition du nombre de ressorts si non manuelle
	i=1 # 1/i donne la proportion de ressorts utilises

	# Changement des rest positions ?
	restPosition = False
	
	# Assignation des parametres de rigidite
	youngModulus = 500
	#if(runFromScript):
		#youngModulus = int(yM)
	poisson = '0.49' 
	springStiffness = '500000'

	# Tolerance
	tolerance =0.03
	
	# Assignation des variable pour le nom des noeuds
	n=1
	n2=n
	baseNomGrille = 'RGrid'
	baseNomObject = 'Liver'
		
	# Mise a l'echelle
	scaled=0 # 0 : pas de scaling, 1 : liver2 2 fois plus grand
	if(scaled==1):
		scale='2 2 2'
	else:
		scale='1 1 1'
		
	########################################################################################################################################
	# Fonctions de base pour creation de la scene
	#
	def createSetOfPoints(self,i):
		l=self.nbMeshPoints
		n=0
		self.setOfPoints=''
		while (n<l):
			if (n%i==0):
				self.setOfPoints+=str(n)+' '
			n+=1	

	def createSpringList(self):
		l=self.nbMeshPoints
		n=0
		self.setOfSprings=''
		sop=str.split(self.setOfPoints)
		while (n<l):
			if (isIn(n,sop)):
				self.setOfSprings+=str(n)+' '+str((n+1)%l)+' '+self.springStiffness+' 1 0 '
			else:
				self.setOfSprings+=str(n)+' '+str((n+1)%l)+' 0 1 0 '
					
			n+=1

	########################################################################################################################################
	# Fonctions pour creation de la scene
	#
		
	def createGlobalStuff(self,node):
		# scene global stuff
		node.createObject('VisualStyle', displayFlags='showInteractionForceFields')
		node.createObject('EulerImplicitSolver', name='cg_odesolver', printLog='0', vdamping='5.0')
		node.createObject('CGLinearSolver', name='linear solver', iterations='100', tolerance='1e-09', threshold='1e-09' )
		node.findData('dt').value=0.02
		node.findData('gravity').value='0 -10 0'

		# Repere pour tester la modification des rest_position
		self.repere=node.createObject('MechanicalObject',name='repere', template='Rigid', position='0000001')
		node.createObject('UniformMass', name='massRepere', totalmass='1')
		
	def createLiver2(self,node):	
		liver2Node = node.createChild('Liver2')
		liver2Node.createObject('MechanicalObject',name='dofs2')
		liver2Node.createObject('UniformMass', name='mass2', totalmass='1')
		if(self.scaled==1):
			liver2Node.createObject('RegularGrid', nx=self.nVal1Char, ny=self.nVal1Char, nz=self.nVal1Char, xmin='-10', xmax='5', ymin='10', ymax='20.5', zmin='-4', zmax='6', name='RG2') # Scaled (x2)
		else:
			liver2Node.createObject('RegularGrid', nx=self.nVal1Char, ny=self.nVal1Char, nz=self.nVal1Char, xmin='-5', xmax='2.5', ymin='10', ymax='15.5', zmin='-2', zmax='3', name='RG2') # Not scaled
		
		surf2Node = liver2Node.createChild('Surface')
		surf2Node.createObject('MeshObjLoader', name='loader2', filename=self.nomMesh+'2.obj')
		surf2Node.createObject('MeshTopology', src='@loader2', name='topo2')
		self.pObject2 = surf2Node.createObject('MechanicalObject', src='@loader2', name='points2', dy='10', scale3d=self.scale)
		surf2Node.createObject('Triangle', name='collisionTriangles')
		surf2Node.createObject('BarycentricMapping')	

		visu2Node = liver2Node.createChild('Visu')
		visu2Node.createObject('OglModel', name='Visual2', fileMesh=self.nomMesh+'2.obj', color='blue', dy='10', scale3d=self.scale)
		visu2Node.createObject('BarycentricMapping', object1='../..', object2='Visual2')	

		liver2Node.createObject('FixedConstraint', name='FixedConstraint', fixAll='1') 
		
	def createGridsAndLivers(self,node):	
		if(self.multi):
			self.youngModulus*=math.pow( 10, self.nGrid )
			print "youngModulus", self.youngModulus
		
		while (self.n <= self.nGrid):
			self.nChar = str(self.n)
			nom = self.baseNomGrille+self.nChar
			self.nChar2 = str(self.nVal1*self.n)
			
			if (self.multi and self.n%2==0):
				self.youngModulus/=100
				print "youngModulus actu", self.youngModulus
			#print 'creation grille',nom
			node.createChild(nom)
			node.getChild(nom).createObject('RegularGrid', nx=self.nChar2, ny=self.nChar2, nz=self.nChar2, xmin='-5', xmax='2.5', ymin='0', ymax='5.5', zmin='-2', zmax='3', name=nom)
			node.getChild(nom).createObject('MechanicalObject', name=nom+'dofsG')
			node.getChild(nom).createObject('UniformMass', name=nom+'mass', totalmass='1')
			node.getChild(nom).createObject('TetrahedronFEMForceField', name=nom+'LiverFEM', youngModulus=str(self.youngModulus), poissonRatio=self.poisson, method='svd')
			
															
			if(self.n < self.nGrid):
				self.n2=10*(self.n)+self.n+1
				self.nChar = str(self.n2)
				nom2 = self.baseNomGrille+self.nChar
				self.nChar2 = str(self.nVal1*(self.n+1))
				#print 'creation grille',nom2
				node.createChild(nom2)
				node.getChild(nom2).createObject('RegularGrid', nx=self.nChar2, ny=self.nChar2, nz=self.nChar2, xmin='-5', xmax='2.5', ymin='0', ymax='5.5', zmin='-2', zmax='3', name=nom2)	
				node.getChild(nom2).createObject('MechanicalObject', name=nom2+'dofsG')
				node.getChild(nom2).createObject('UniformMass', name=nom2+'mass', totalmass='1')
				node.getChild(nom2).createObject('BarycentricMapping', input='@../'+nom+'/'+nom+'dofsG', output='@'+nom2+'dofsG')


			self.nChar = str(self.n)
			nom3 = self.baseNomObject+'1'+self.nChar
			#print 'creation liver',nom3
			node.createChild(nom3)
			node.getChild(nom3).createObject('MeshObjLoader', name='loader'+nom3, filename=self.nomMesh+'1.obj')
			node.getChild(nom3).createObject('MeshTopology', src='@loader'+nom3, name='topo'+nom3)
			node.getChild(nom3).createObject('MechanicalObject', src='@loader'+nom3, name='points'+nom3)
			node.getChild(nom3).createObject('BarycentricMapping', input='@../'+nom+'/'+nom+'dofsG', output='@points'+nom3)
			node.getChild(nom3).createObject('Point', name='collisionPoints')


			node.getChild(nom3).createChild('Visu')
			node.getChild(nom3).getChild('Visu').createObject('OglModel', name='Visual11'+nom3, fileMesh=self.nomMesh+'1.obj', color='gray')
			node.getChild(nom3).getChild('Visu').createObject('BarycentricMapping', object1='@../points'+nom3, object2='@Visual11'+nom3)

			#node.getChild(nom3).createObject('EvalSurfaceDistance', name='evalDist', object1='@points'+nom3, object2='@../Liver2/Surface/points2', period='0.1', draw='True', filename='Output/DistOutput'+nom3+'.txt', listening='True')


			# Force field liant des points des 2 liver 2 a 2 (lie les points des mesh, pas ceux des grilles)
			#print "nom3 no",self.n," = ", nom3
			node.getChild(nom3).createObject('AdaptativeSpringForceField', name='link'+nom3, object1='points'+nom3 , object2='../Liver2/Surface/points2', spring=self.setOfSprings) 
	

			self.n+=1 
			
	def createSpringArg(self):
		# Nombre de points dans les mesh a recaler 
		self.nbMeshPoints = len(self.pObject2.findData('position').value)
		
		# Definition de la liste des points relies par des ressorts
		if (self.manual):
			#self.setOfPoints='14 39 70 113' 
			#self.setOfPoints='85 86 89 92 93 94 96 100 101 104 105 108 109 110' # Points localises sur la surface "visible"
			#self.setOfPoints='62 84 85 86 89 92 93 94 96 100 101 104 105 108 109 110 113 118' # Points localises sur la surface "visible"
			self.setOfPoints='46 48 49 50 51 57 58 59 64 65 93 94 118 119 120 159 160 161 164 165 166 168 255 257 259' # Points localises sur la surface "visible"
		else:
			self.createSetOfPoints(self.i)
		
		# Creation des ressorts
		self.createSpringList()
		#print 'liste ressorts : ', self.setOfSprings
		
		
		

	########################################################################################################################################
	#							Node creation                                                    
	########################################################################################################################################
	def createScene(self,node):
		
		self.createGlobalStuff(node)
		
		# Creation du liver cible ######################################################################################################
		self.createLiver2(node)
		
		# Creation de la liste des arguments a transmettre au composant spring #########################################################
		self.createSpringArg()
		
		# Creation des grilles et des livers ###########################################################################################
		self.createGridsAndLivers(node)
		
		print 'i = ',self.i
		print 'youngModulus = ', self.youngModulus 
		print 'poisson = ', self.poisson
		print 'self.springStiffness = ', self.springStiffness	
		
		return 0




	#############################################################################################################
	# Autres fonctions pour initialisation
	#

	def createNomFichier(self):
		self.nomFichier='applications-dev/plugins/Registration-dev/examples/Output/Avec_point_dans_foie/test_matching2'
		if (self.multi):
			self.nomFichier+='MG_'
		else:
			self.nomFichier+='UG_'

		if (self.restPosition):
			self.nomFichier+='R_'
			
		if (self.multi):
			self.nomFichier+='1000_100_10_'

		self.nomFichier+=str(len(str.split(self.setOfPoints)))+'_'+str(self.youngModulus)+'_'+self.springStiffness+'_'+self.poisson+'_tolerance' + str(self.tolerance)

		if (self.manual):
			self.nomFichier+='_m'
			
	def createIndicesToFollow(self):
			l=self.nbMeshPoints
			n=0
			indicesToFolow=''
			while (n<l):
				indicesToFolow+=str(n)+' '
				n+=1	
			self.indicesToFolow=str.split(indicesToFolow)
			n=0
			while (n<self.nbMeshPoints):
				self.indicesToFolow[n]=int(self.indicesToFolow[n])
				n+=1
				
	def initializationObjects(self,node):
		self.createNomFichier()
		self.fichier = open(self.nomFichier, "w")
		self.rootNode = node.getRoot()
		self.n=1
		
		# Def des liver a utiliser (un bouge l'autre recup les position une fois palier atteint)
		nom3 = self.baseNomObject+'11'
		self.pObject11 = self.rootNode.getChild(nom3).getObject('points'+nom3)
		self.spring = self.rootNode.getObject('link'+nom3)
		if (self.multi):
			nom3 = self.baseNomObject+'12'
			self.pObject12 = self.rootNode.getChild(nom3).getObject('points'+nom3)
			# Def des Rgrid a utiliser (une bouge l'autre recup les position une fois palier atteint)
			nom = self.baseNomGrille+'12'
			self.rGObject1 = self.rootNode.getChild(nom).getObject(nom+'dofsG')
			nom = self.baseNomGrille+'2'
			self.rGObject2 = self.rootNode.getChild(nom).getObject(nom+'dofsG')
			
		self.count 	= 	0
		self.count2 	= 	0
		self.modSpCount = 	0
		self.error 	= 	100000.0
		self.errorMin	=	1000.0 	# Pour determiner la tolerance a utiliser si celle utilisee est trop grande
		self.testMin	=	0
		self.errorMax	=	0.0001 	# Pour determiner la tolerance a utiliser si celle utilisee est trop grande
		self.testMax	=	0
		self.deltaErr	=	1000.0 	# Pour determiner la tolerance a utiliser si celle utilisee est trop grande
		self.test	=	0
		self.lenthPalier = 	100
		self.createIndicesToFollow()
				
	def stepZero(self):
		n = 1

		self.fichier.write('# step error error_per_pt deltaerr errPtAdded')
		n = 1
		while (n < self.nGrid):
			n += 1
			self.rootNode.getChild(self.baseNomGrille+str(n)).active = False
			if (n < self.nGrid):
				self.rootNode.getChild(self.baseNomGrille+str(n)+str(n+1)).active = False
			self.rootNode.getChild(self.baseNomObject+'1'+str(n)).active = False
						
	#############################################################################################################
	# Fonctions pour modifier les parametres du probleme
	#

	def changeDimension(self):
		if (self.n < self.nGrid):
			nom3 = self.baseNomObject+'1'+str(self.n+1)
			
			self.rootNode.getChild(self.baseNomGrille+str(self.n+1)).active = True
			self.rootNode.getChild(self.baseNomObject+'1'+str(self.n+1)).active = True
			if (self.n < self.nGrid-1):
				self.rootNode.getChild(self.baseNomGrille+str(self.n+1)+str(self.n+2)).active = True

			self.rGObject2.findData('position').value = self.rGObject1.findData('position').value
			self.pObject12.findData('position').value = self.pObject11.findData('position').value
			
			#print "nom3 utilise pour changer de spring = ", nom3
			self.spring = self.rootNode.getChild(nom3).getObject('link'+nom3)
			

			self.rootNode.getChild(self.baseNomObject+'1'+str(self.n)).active = False # Desactivation du Liver utilise precedement				
			self.rootNode.getChild(self.baseNomGrille+str(self.n)).active = False # Desactivation de la grille qui controlait la transformation
			self.rootNode.getChild(self.baseNomGrille+str(self.n)).active = False # Desactivation de la grille qui suivait la grille precedente (utile pour les positions uniquement)
			
			
			
		
			
		self.n += 1 # Laisser cette ligne ici car self.n controle les noms des objects 


		if (self.n <= self.nGrid):
			# Changement de livers
			self.pObject11 = self.pObject12

		if (self.n < self.nGrid):
			# Changement de livers
			nom3 = self.baseNomObject+'1'+str(self.n+1)
			#print 'changement de liver : self.n = ',self.n, 'nom3 = ', nom3
			self.pObject12 = self.rootNode.getChild(nom3).getObject('points'+nom3)

			# Changement de grilles	
			nom = self.baseNomGrille+str(self.n)+str(self.n+1)
			#print 'changement de grille : self.n = ',self.n, 'nom = ', nom
			self.rGObject1 = self.rootNode.getChild(nom).getObject(nom+'dofsG')					
			nom = self.baseNomGrille+str(self.n+1)
			#print 'changement de grille : self.n = ',self.n, 'nom = ', nom
			self.rGObject2 = self.rootNode.getChild(nom).getObject(nom+'dofsG')


	def changeDimensionReinit(self):
		if (self.n < self.nGrid):
			nom3 = self.baseNomObject+'1'+str(self.n+1)
			self.rootNode.getChild(self.baseNomGrille+str(self.n+1)).active = True
			self.rootNode.getChild(self.baseNomObject+'1'+str(self.n+1)).active = True
			if (self.n < self.nGrid-1):
				self.rootNode.getChild(self.baseNomGrille+str(self.n+1)+str(self.n+2)).active = True
			self.spring = self.rootNode.getChild(nom3).getObject('link'+nom3)

			#print self.rGObject2.findData('rest_position').value
			self.rGObject2.findData('position').value = self.rGObject1.findData('position').value
			self.pObject12.findData('position').value = self.pObject11.findData('position').value
			self.rGObject2.findData('rest_position').value = self.rGObject1.findData('position').value
			self.pObject12.findData('rest_position').value = self.pObject11.findData('position').value
			
			self.rGObject2.init()
			self.pObject12.init()
			#print self.rGObject2.findData('rest_position').value

			self.rootNode.getChild(self.baseNomObject+'1'+str(self.n)).active = False # Desactivation du Liver utilise precedement				
			self.rootNode.getChild(self.baseNomGrille+str(self.n)).active = False # Desactivation de la grille qui controlait la transformation
			self.rootNode.getChild(self.baseNomGrille+str(self.n)).active = False # Desactivation de la grille qui suivait la grille precedente (utile pour les positions uniquement)
					
		self.n += 1 # Laisser cette ligne ici car self.n controle les noms des objects 

		if (self.n <= self.nGrid):
			# Reinit du force field				
			nom = self.baseNomGrille+str(self.n)
			self.FFObject = self.rootNode.getChild(nom).getObject(nom+'LiverFEM')
			self.FFObject.reinit()	
			self.FFObject.init()	
			self.FFObject.bwdInit()				

		if (self.n <= self.nGrid):
			# Changement de livers
			self.pObject11 = self.pObject12

		if (self.n < self.nGrid):
			# Changement de livers
			nom3 = self.baseNomObject+'1'+str(self.n+1)
			#print 'changement de liver : self.n = ',self.n, 'nom3 = ', nom3
			self.pObject12 = self.rootNode.getChild(nom3).getObject('points'+nom3)

			# Changement de grilles	
			nom = self.baseNomGrille+str(self.n)+str(self.n+1)
			#print 'changement de grille : self.n = ',self.n, 'nom = ', nom
			self.rGObject1 = self.rootNode.getChild(nom).getObject(nom+'dofsG')					
			nom = self.baseNomGrille+str(self.n+1)
			#print 'changement de grille : self.n = ',self.n, 'nom = ', nom
			self.rGObject2 = self.rootNode.getChild(nom).getObject(nom+'dofsG')

		
		return 0
