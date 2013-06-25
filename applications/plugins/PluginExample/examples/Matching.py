import Sofa
import math
import os
import FC

from decimal import Decimal


############################################################################################
# Recalage Matching
############################################################################################


class Matching(FC.StatAndFC):
	
	############################"  
	# Definition des parametres :
	############################"

	# Garder le matching de depart ?
	keepMatching = False
	prctAppa=-100


	#############################################################################################################
	# Fonctions pour modifier les parametres du probleme
	#

	def changeCoresPoints(self,cat):
		nom3 = self.baseNomObject+'1'+str(self.n)
		self.spring = self.rootNode.getChild(nom3).getObject('link'+nom3)
		#print "changeCoresPoints "
		sop=str.split(self.setOfPoints)
		pourcentAppa=0.0
		lowStiffness=0
		if(cat==1 or cat==2):
			dist=0	
			jMin = [0] * (len(sop))
		elif(cat==3 or cat==4):
			dist1=0	
			dist2=0	
			jMin1 = [0] * (len(sop))
			jMin2 = [0] * (len(sop))
		#print "v = self.spring.spring"
		v = self.spring.spring

		# trouver le plus proche voisin de chaque point du mesh (corresspondance point a point et pas point a surface pour le moment)
		for i in range (len(sop)): 
			if(cat==1 or cat==2):
				distMin=1000.0
			elif(cat==3 or cat==4):
				distMin1=1000.0
				distMin2=1000.0	
			for j in range (len(sop)): 	



				if(cat==1):
					dist=self.norm2(i,j) #attention ici i est sur le liver1 et j sur le liver 2
				elif(cat==2):
					dist=self.norm2(j,i) #attention ici i est sur le liver2 et j sur le liver 1
				elif(cat==3 or cat==4):
					dist1=self.norm2(i,j)
					dist2=self.norm2(j,i)

				if(cat==1 or cat==2):
					if (dist<distMin):
						distMin=dist
						jMin[i]=j

				elif(cat==3 or cat==4):
					if (dist1<distMin1):
						distMin1=dist1
						jMin1[i]=j
					if (dist2<distMin2):
						distMin2=dist2
						jMin2[i]=j


			if(cat==1):
				v[i][0].Index2 = jMin[i] # Attention ici on suppose que i sur liver1 et donc jmin sur liver 2		
				#print "Nouvel index correspondant pour le point ",v[i][0].Index1," : ", v[i][0].Index2, "dist = ",distMin
			elif(cat==2):
				v[i][0].Index2 = i # Attention ici on suppose que i sur liver2 			
				v[i][0].Index1 = jMin[i] # Attention ici on suppose que i sur liver2 et donc jmin sur liver 1		
				#print "Nouvel index correspondant pour le point ",v[i][0].Index2," : ", v[i][0].Index1, "dist = ",distMin
			elif(cat==3):
				v[i][0].Index2 = jMin1[i] # Attention ici on suppose que i sur liver1 donc jmin sur liver 2
				#print "Nouvel index correspondant pour le point ",v[i][0].Index1," : ", v[i][0].Index2, "dist = ",distMin
			elif(cat==4):
				v[i][0].Index2 = i # Attention ici on suppose que i sur liver2 			
				v[i][0].Index1 = jMin2[i] # Attention ici on suppose que i sur liver2 et donc jmin sur liver 1
				#print "Nouvel index correspondant pour le point ",v[i][0].Index2," : ", v[i][0].Index1, "dist = ",distMin		
			#print " "
			
			
			if(cat==1 or cat==2):
				if (i==jMin[i]):
					pourcentAppa+=1
			elif(cat==3 or cat==4):
				if (i==jMin1[i]):
					pourcentAppa+=1
					

		if(cat==3 or cat==4):
			for k in range (len(sop)): 
									
				if(jMin2[jMin1[k]]==k):
					v[k][0].Ks=int(self.springStiffness)*1.0
				
				else:
					lowStiffness+=1
					v[k][0].Ks=int(self.springStiffness)*0.5

			
		pourcentAppa/=(len(sop))
		pourcentAppa*=100
		print "pourcentAppa = ", pourcentAppa
		
		if(cat==3 or cat==4):
			print "Nonbre de points pour lesquels la correspondance n'est pas symetrique : ",lowStiffness
			
		if(abs(pourcentAppa-self.prctAppa)<1):
			if (self.multi):
				if(self.restPosition):
					print "######################## Changement de grille ########################"
					self.changeDimensionReinit()
							
				else:
					print "######################## Changement de grille ########################"
					self.changeDimension() 
							
			else:
				self.n -= 1 
			
		self.prctAppa=pourcentAppa
		self.spring.spring = v

		
		return 0