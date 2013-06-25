import Sofa
import math
import os
import FC
import Matching

from decimal import Decimal


############################################################################################
# Recalage Multi grille
############################################################################################


############################################################################################
# following defs are used later in the script
############################################################################################

class Recalage(Matching.Matching):
	
############################"  
# Definition des parametres :
############################"

	
	

	def __init__(self,multi,restPosition,keepMatch,updateStiffness,runFromScript):
		self.multi=multi
		self.restPosition=restPosition
		self.keepMatching=keepMatch
		self.updateStiffness=updateStiffness
		self.runFromScript=runFromScript
		
		
	def process(self,step,total_time):
		#print "self.n <= self.nGrid : ",self.n," <= ",self.nGrid
		if (self.n <= self.nGrid):
			if (step == 1):
				self.stepZero()

			#print "debut update erreur"
			self.updateErr()			
			#print "fin update erreur"
			
			#print "debut updateMinMaxForTolerance"
			self.updateMinMaxForTolerance()
			#print "fin updateMinMaxForTolerance"
			
			#print "debut updateCountForPalier"
			self.updateCountForPalier()
			#print "fin updateCountForPalier"
			
			#print "count = ",self.count
			
			## Pour modifier la tolerance si les oscillation sont plus importante que la tolerance 			
			#print "debut testTolerance"
			self.testTolerance()
			#print "fin testTolerance"

			
			## Si on atteit le palier on l'indique, on arrete l'animation, on modifie les proprietes de l'objet a recaler et de la tolerance et on remet le compteur a zero
			if (self.count>self.lenthPalier):
				print 'Palier atteint t = ', total_time, ', step = ', step, ', error = ', self.err, ', error par pt = ', self.err/self.nbMeshPoints, ', deltaerr = ', abs(self.err-self.error),', tolerance = ',self.tolerance
				errMax=10 
				iMax=15

				self.test = 0
				self.error = self.err	
				self.count = 0
				
				if (self.multi and self.keepMatching):
					if(self.restPosition):
						self.changeDimensionReinit()
						#print "multi et garder matching reinit"
					else:
						self.changeDimension() 
						#print "multi et garder matching"
						
				elif(self.multi==False and self.keepMatching):
					#print "non multi et garder matching"
					self.n += 1 
				else:
					#print "ne pas garder matching"
					self.changeCoresPoints(1)
					
				self.statErrByPoint()

				#if(self.auto==False):
					#self.rootNode.getRootContext().animate = False
				
				print "HausdorffDiscrete = ",self.hausdorffDiscret()
				print " "
				print " "
			
		else:
			self.rootNode.getRootContext().animate = False
			print "Plus de grilles pour raffinement supplementaire. Arret de l'animation."
			self.fichier.close()
			if(self.runFromScript):
				quit()


	#############################################################################################################
	# Fonctions pour suivi de l'erreur
	#


	def norm1(self,i,j):
		#print "in norm1"
		err = 0
		for k in range (3) : # aller jusqu'a 3 car sinon on ne prend pas les coordonees z (k n'est jamais egal a 2 )
			positionValues1 = self.pObject11.findData('position').value[self.indicesToFolow[i]][k]
			positionValues2 = self.pObject2.findData('position').value[self.indicesToFolow[j]][k]
			err += abs(positionValues1-positionValues2)
		return err

	def norm2(self,i,j):
		#print "in norm2"
		norm = 0
		for k in range (3) : # aller jusqu'a 3 car sinon on ne prend pas les coordonees z (k n'est jamais egal a 2 )ts
			positionValues1 = self.pObject11.findData('position').value[self.indicesToFolow[i]][k]
			positionValues2 = self.pObject2.findData('position').value[self.indicesToFolow[j]][k]
			norm += (positionValues1-positionValues2)*(positionValues1-positionValues2)
		norm=math.sqrt(norm)
		return norm

	def updateErr(self):
		#print "in update erreur"
		self.err = 0

		# Addition de toutes les differences de position des points que l'on suit 
		for i in range (self.nbMeshPoints) :
			self.err+=self.norm2(i,i)
		#j=self.nbMeshPoints-1

		return 0

	def updateCountForPalier(self):
		#print "in updatecountforpalier"

		# Test sur la difference entre l'erreur actuelle et l'erreur a l'iteration precedente
		if (abs(self.err-self.error)<self.tolerance):# Si on reste en dessous du seuil de tolerance	
			self.count += 1 # On incremente le compteur si la diference est inferieure a un certain seuil

		elif (abs(self.err-self.error)>=self.tolerance):# Si on depase le seuil de tolerance				
			if(self.count>=1):
				# on prend la moyenne entre l'erreur max et l'erreur min trouvee depuis la premiere fois que l'erreur a augmente (utile si la valeur de l'erreur oscille)
				self.error = (self.errorMin + self.errorMax)/2
			else:
				self.error = self.err	
					
			self.count = 0



	#############################################################################################################
	# Fonctions pour update eventuel de la tolerance 
	#
	def updateMinMaxForTolerance(self):
		#print "in update minmaxfortolerance"
		if ((self.err-self.error)>0 and self.test<self.lenthPalier/5): # detecte si l'erreur remonte un certain nombre de fois (permet bon update d'errorMax au changement de palier)
			self.test+=1			
			if(self.test==self.lenthPalier/5):
				self.errorMin=self.err
				self.errorMax=self.err
				self.testMin=0
				self.testMax=0
					
		if (self.errorMin>=self.err or self.testMin>self.lenthPalier*1.5): # reset errorMin si pas de convergence au bout de plus d'un demi palier (utile si l'erreur etait trop descendue apres changement de palier)
			self.errorMin=self.err
			self.testMin=0
		else:
			self.testMin+=1

		if (self.errorMax<=self.err or self.testMax>self.lenthPalier*1.5): # reset errorMax si pas de convergence au bout de plus d'un demi palier (utile si l'erreur etait trop montee apres changement de palier)
			self.errorMax=self.err
			self.testMax=0
		else:
			self.testMax+=1

	def testTolerance(self):
		#print "in testtolerance"
		if (self.errorMax - self.errorMin > self.tolerance*2 and self.test>=self.lenthPalier/5):
			self.test+=1
			if (self.test>=self.lenthPalier/5+self.lenthPalier*4):
				print "Attention tolerance trop faible. Modification de la tolerance : ", (math.floor((self.errorMax - self.errorMin)*100/2)+1)/100
				self.tolerance=(math.floor((self.errorMax - self.errorMin)*100/2)+1)/100

		



