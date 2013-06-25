import Sofa
import math
import os
import multiGrid

from decimal import Decimal


############################################################################################
# Recalage FC
############################################################################################

def maximum(tab):
	l=len(tab)
	maxi=0
	
	for i in range (l) :	
		if(maxi<tab[i]):
		  maxi=tab[i]
		  
	return maxi

class StatAndFC(multiGrid.MultiGrid):
	
############################"  
# Definition des parametres :
############################"

	updateStiffness=True
	modSpCount=0


#############################################################################################################
# Fonctions pour calculs statistiques
#


	def statPointAdded(self):
		# Addition de tous les dx+dy+dz par rapport a la position du point que l'on suit 
		i = len(self.indicesToFolow)-1
		print norm2(self,i,i)
		return 0

	def dxdydz(self,i):
		coord=['dx','dy','dz']
		for j in range (3) : # aller jusqu'a 3 car sinon on ne prend pas les coordonees z (j n'est jamais egal a 2 )
			positionValues1 = self.pObject11.findData('position').value[self.indicesToFolow[i]][j]
			positionValues2 = self.pObject2.findData('position').value[self.indicesToFolow[i]][j]
			err = (positionValues1-positionValues2)
			
			print coord[j], ' = ', err
		return 0

	def dz(self,i):
		
		j=2 
		positionValues1 = self.pObject11.findData('position').value[self.indicesToFolow[i]][j]
		positionValues2 = self.pObject2.findData('position').value[self.indicesToFolow[i]][j]
		err = (positionValues1-positionValues2)
			
		#print 'dz = ', err

		if(err<0):
			return -1
		return 1

	def statErrByPoint(self):
		self.variance = 0

		self.errMax = 0
		self.iMax = 0
		
		self.errMin = self.err
		self.iMin = 0
		
		self.tErr=[0]*self.nbMeshPoints

		self.err/=self.nbMeshPoints

		# Addition de tous les dx+dy+dz par rapport aux position de reference de chaque point que l'on suit 
		for i in range (self.nbMeshPoints) :	
			#print self.indicesToFolow[i], " : ", norm1(self,i)
			self.tErr[i]=self.norm2(i,i)
			self.variance+=(self.tErr[i]-self.err)*(self.tErr[i]-self.err)
			if(self.tErr[i]<self.errMin):
				self.errMin=self.tErr[i]
				self.iMin=i
			if(self.tErr[i]>self.errMax):
				self.errMax=self.tErr[i]
				self.iMax=i
		self.variance/=self.nbMeshPoints
		print "Variance = ", self.variance, "ErrMin ( ", self.errMin , " ) atteinte pour i = ",self.indicesToFolow[self.iMin], "ErrMax ( ", self.errMax , " ) atteinte pour i = ",self.indicesToFolow[self.iMax]
		self.statErrHisto()
		self.changeRessort3()
		return 0


	def statErrHisto(self):
		sepNumber=10
		self.histo = [0] * (sepNumber) 
		self.histoc = [0] * (sepNumber) 
		self.delta=(self.errMax-self.errMin)/sepNumber
		# Addition de tous les dx+dy+dz par rapport aux position de reference de chaque point que l'on suit 
		for i in range (len(self.indicesToFolow)) :
			for j in range (sepNumber):	
				if(self.norm2(i,i)<=self.errMin+self.delta*(1+j)):
					self.histoc[j]+=1
					if(self.errMin+self.delta*(j)<self.norm2(i,i)):
						self.histo[j]+=1		
		
		print "Histo = ", self.histo
		print "Histoc = ", self.histoc
		return 0

	def hausdorffDiscret(self):
		
		#print "changeCoresPoints "
		sop=str.split(self.setOfPoints)
		self.distMin12= [1000.0] * (len(sop))
		self.distMin21= [1000.0] * (len(sop))
		
		#print "v = self.spring.spring"
		v = self.spring.spring
		#print "1er appel de changeCoresPoints "
		# trouver le plus proche voisin de chaque point du mesh (corresspondance point a point et pas point a surface pour le moment)
		for i in range ((len(sop))): 
		
			for j in range ((len(sop))): 
			
				self.dist12=self.norm2(i,j)			
				if (self.dist12<self.distMin12[i]):
					#print dist, "<", distMin
					self.distMin12[i]=self.dist12
					
					
				
				self.dist21=self.norm2(j,i)
				if (self.dist21<self.distMin21[i]):
					#print dist, "<", distMin
					self.distMin21[i]=self.dist21
					
		self.dist=self.distMin12+self.distMin21
		
		return maximum(self.dist)


	#############################################################################################################
	# Fonctions pour modifier les parametres du probleme
	#

	def changeRessort1(self):
		
		#print 'ressort1= ', self.spring.findData('name').value 
		
		#print 'ressort2 = ', str(self.spring.findData('spring').value)
		v = self.spring.spring

		#v[1]=['15 15 '+self.springStiffness+' 1 0 ']
		for i in range(self.nbMeshPoints):
			if(self.tErr[i]>(self.err+math.sqrt(self.variance)*0.5)):
				v[i][0].Ks = int(self.springStiffness)*(1.0+self.tErr[i])
				print 'Stiffness ressort[',i,'] = ', str(v[i][0].Ks)
			else:
				v[i][0].Ks = math.ceil(v[i][0].Ks /(2+ (self.err-self.tErr[i])))
				if (v[i][0].Ks> int(self.springStiffness)/2):
					print '                                            Stiffness ressort[',i,'] = ', str(v[i][0].Ks)
		self.spring.spring = v
		
		print ' '
		print 'err = ', self.err
		
		print '\n\n\n\n'
		self.n -= 1 

		if (self.modSpCount==0):
			changeYoungModulus(self)

		
		self.modSpCount+=1
		
		return 0


	def changeRessort2(self):
		
		#print 'ressort1= ', self.spring.findData('name').value 
		
		#print 'ressort2 = ', str(self.spring.findData('spring').value)
		v = self.spring.spring

		#v[1]=['15 15 '+self.springStiffness+' 1 0 ']
		for i in range(self.nbMeshPoints):
			if(self.tErr[i]>(self.err+math.sqrt(variance)*0.5)):
				v[i][0].Ks = int(self.springStiffness)*(1.0+self.tErr[i])
				print 'Stiffness ressort[',i,'] = ', str(v[i][0].Ks)
			else:
				v[i][0].Ks = math.ceil(v[i][0].Ks /(2+ (self.err-self.tErr[i])))
				if (v[i][0].Ks> int(self.springStiffness)/2):
					print '                                            Stiffness ressort[',i,'] = ', str(v[i][0].Ks)
		self.spring.spring = v
		
		print ' '
		print 'err = ', self.err
		
		print '\n\n\n\n'
		self.n -= 1 

		if (self.modSpCount==0):
			changeYoungModulus(self)

		
		self.modSpCount+=1
		
		return 0


	def changeRessort3(self):
		
		#print 'ressort1= ', self.spring.findData('name').value 
		
		#print 'ressort2 = ', str(self.spring.findData('spring').value)
		v = self.spring.spring

		
		for i in range(self.nbMeshPoints):
			
			if(self.modSpCount==5):
				print "Modification des ressorts 5eme iteration"
				if (v[i][0].Ks< int(self.springStiffness)):
					v[i][0].Ks= math.floor(100*math.exp(2*(self.tErr[i]/self.err-1)))
				else:
					v[i][0].Ks = math.floor(int(self.springStiffness)*math.exp(2*(self.tErr[i]/self.err-1)))
			else:
				v[i][0].Ks = math.ceil(v[i][0].Ks*math.exp(2*(self.tErr[i]/self.err-1))) # Attention rigidification des points ou l'erreur est faible
			#v[i][0].Ks = math.ceil(v[i][0].Ks*math.exp(pow((tErr[i]/err-1),3)))
			#v[i][0].Ks = math.ceil(v[i][0].Ks*math.exp((tErr[i]/err)-1))
			#print 'math.exp((tErr[i]/err)-1) = ',math.exp((tErr[i]/err)-1)
			if (v[i][0].Ks> int(self.springStiffness)):
				print '                                            Stiffness ressort[',i,'] = ', str(v[i][0].Ks)	
			else:
				print 'Stiffness ressort[',i,'] = ', str(v[i][0].Ks)
		
			
			

		self.spring.spring = v
		
		print ' '
		print 'err = ', self.err
		
		print '\n\n\n\n'
		#self.n -= 1 
		
		self.modSpCount+=1
		
		return 0


	def changeYoungModulus(self):
		# Reinit du force field				
		nom = self.baseNomGrille+str(self.n)
		FFObject = self.rootNode.getChild(nom).getObject(nom+'LiverFEM')
		FFObject.findData('youngModulus').value /= 10000
		FFObject.reinit()	
		FFObject.init()	
		FFObject.bwdInit()	
		



	####################################################################"
		#print 'ressort1= ', self.spring.findData('name').value 

		#print 'ressort2 = ', str(self.spring.findData('spring').value)
		#v = self.spring.findData('spring').value
		# ou plus simplement:
		#v = self.spring.spring

		##v[1]=['15 15 '+self.springStiffness+' 1 0 ']
		#print 'set index1'
		#print len(v[0][0])
		#v[0][0].Index1 = 15
		#print 'set index2'
		#v[0][0].Index2 = 15
		#print 'set ks'
		#v[0][0].Ks = int(self.springStiffness)*0.50
		#print 'set kd'
		#v[0][0].Kd = 1
		#print 'set l'
		#v[0][0].L = 0
		#self.spring.findData('spring').value = v	
		## ou plus simplement:
		##self.spring.spring = v
		#print 'v = ', str(v)
		#print 'ressort3 = ', str(self.spring.findData('spring').value[0][1].Ks)
	########################################################################


