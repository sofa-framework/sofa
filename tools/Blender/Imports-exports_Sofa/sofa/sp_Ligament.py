# -*- coding: iso8859-15 -*- 

import Blender
from Blender import Curve, Draw

from math import pi
from numpy import matrix
from scipy.linalg import inv, det, eig

import sp_Xstring
reload( sp_Xstring )
from sp_Xstring import *

import sp_XML
reload( sp_XML )
from sp_XML import *

import sp_Math3D
reload( sp_Math3D )
from sp_Math3D import *


class Ligament:
	
	def setListeNomsOs( self, listeNomsOs ):
		self.listeNomsOs = listeNomsOs
		#print "Longueur de la liste : %d "%(len(lst
		for strNom in self.listeNomsOs:
			print "%s"%(strNom)
	
	
	def setParametresCourbe( self, courbe ):
		self.strNomObjet = courbe.getName()
		self.posLineaire = courbe.getLocation()
		self.curve = courbe.getData()
		self.posAngulaire = self.curve.getRot()
		self.dimensions = self.curve.getSize()
	
	
	def setParametresXML( self, strNomFichierXML ):
		self.strNomFichierXML = strNomFichierXML
	
	
	def mettre_fichier_XML_a_jour( self ):
		fxXML = XML()
	
	
	def afficher( self ):
		print "Nom objet : \"%s\""%( self.strNomObjet )
		print "Fic. XML  : \"%s\""%( self.strNomFichierXML )
		print self.posLineaire
		print self.posAngulaire
		print "- Taille x=%f y=%f z=%f"%(self.dimensions[0], self.dimensions[1], self.dimensions[2])
		nbCourbes = self.curve.getNumCurves()
		if nbCourbes != 1:
			strM = "Attention|La courbe \"%s\" ne devrait contenir qu'une seule courbe."%(self.strNomObjet)
			Draw.PupMenu( strM )
		nbPtsCtrl = self.curve.getNumPoints()
		print "- Nb pts de controle : %d"%(nbPtsCtrl)
		print "- Location x=%f y=%f z=%f"%(self.posLineaire[0], self.posLineaire[1], self.posLineaire[2])
		print "- Rotation x=%f y=%f z=%f"%(self.posAngulaire[0], self.posAngulaire[1], self.posAngulaire[2])
		for cur in self.curve:
			print type( cur ), cur
			for point in cur:
				print type( point ), point
		print "\n"
	
	
	def getNomOsLePlusProche( self, pt ):
		# Parcours de la liste des os de la scène et récupération du nom de celui qui est le plus
		#  proche du point reçu
		lst_obj = Blender.Object.Get()
		distMinGlobale = 100000.1
		nIndexSommet = -1
		nIndexSommetRetour = nIndexSommet
		fxMaths = Math3D()
		objTrouve = None
		verbose = 0
		for strNom in self.listeNomsOs:
			# Récupération du maillage de l'os
			if verbose == 1:
				print "Recherche de %s "%(strNom)
			objTrouve = None
			for obj in lst_obj:
				if obj.getName() == strNom and obj.getType() == "Mesh":
					objTrouve = obj
			
			# Calcul de la distance mini entre le point reçu et ce maillage
			if objTrouve != None:
				if verbose == 1:
					print "Objet %s "%(objTrouve.getName())
				ptLocat = objTrouve.getLocation()
				ptRotat = objTrouve.getEuler() 
				
				# Calcul de la matrice de transformation de cet objet
				#Q_rot = fxMaths.calcQuaternionWiki( ptRotat(0), ptRotat(1), ptRotat(2) )
				Q_rot = fxMaths.calcQuaternionWiki( ptRotat[0], ptRotat[1], ptRotat[2] )
				if verbose == 1:
					print "Quaternion de rotation"
					print Q_rot
				
				M_rot = fxMaths.quat_to_matrix( Q_rot )
				if verbose == 1:
					print "Matrice de rotation"
					print M_rot
				
				vT = fxMaths.convertTupleToArray( ptLocat )
				M_tr = fxMaths.calcMatriceTranslation( vT )
				if verbose == 1:
					print "Matrice de translation"
					print M_tr
				
				M = dot(M_tr, M_rot)
				if verbose == 1:
					print "Matrice de transformation"
					print M
				
				distMin = ( (ptLocat[0] - pt[0])**2 + (ptLocat[1] - pt[1])**2 + (ptLocat[2] - pt[2])**2 )
				nIndexSommet = -1
				maillage = objTrouve.getData( mesh = 1 )
				
				for v in maillage.verts:
					ptLoc = array([0.0, 0.0, 0.0, 1.0])
					ptLoc[0] = v.co[0]
					ptLoc[1] = v.co[1]
					ptLoc[2] = v.co[2]
					ptOsGlob = dot(M, ptLoc)
					dist = ( (ptOsGlob[0]-pt[0])**2 + (ptOsGlob[1]-pt[1])**2 + (ptOsGlob[2]-pt[2])**2 )
					if ( dist < distMin ):
						distMin = dist
						nIndexSommet = v.index
				if verbose == 1: 
					print "+ Distance mini os %s sur sommet %d = %f"%(strNom, nIndexSommet, distMin)
				# Verification si c'est le mini
				if distMin < distMinGlobale:
					distMinGlobale = distMin
					strNomRetour = strNom
					nIndexSommetRetour = nIndexSommet
		if verbose == 1: 
			print "%s"%(strNomRetour)
		return strNomRetour, nIndexSommetRetour
		
		
	def calculerPositionPointGlobDansRepereObjet( self, ptGlobal, strNomObj ):
		objTrouve = Blender.Object.Get( strNomObj )
		ptLocat = objTrouve.getLocation()
		ptRotat = objTrouve.getEuler() 
		fxMaths = Math3D()
		Q_rot = fxMaths.calcQuaternionWiki( ptRotat[0], ptRotat[1], ptRotat[2] )
		M_rot = fxMaths.quat_to_matrix( Q_rot )
		vT = fxMaths.convertTupleToArray( ptLocat )
		M_tr = fxMaths.calcMatriceTranslation( vT )
		M = dot(M_tr, M_rot)
		M_inv = inv( M )
		pt_loc = dot(M_inv, ptGlobal)
		return pt_loc


	def mettre_fichier_XML_a_jour( self ):
		fxMaths = Math3D()
		verbose = 1
		print "\n"
		print "Mise a jour du fichier XML pour le ligament %s "%(self.strNomObjet)
		# Recherche de l'os le plus proche pour y accrocher l'extremité du ligament
		# - calcul de la position absolue du point dans la scène
		lstCurv = self.curve[0]
		bzTriple = lstCurv[0]
		if verbose == 1:
			print type(bzTriple), bzTriple
		lst_3_pts = bzTriple.getTriple()
		ptB_bezier_local = lst_3_pts[1] # point d'attache
		ptC_bezier_local = lst_3_pts[2]
		bzTriple = lstCurv[1]
		lst_3_pts = bzTriple.getTriple()
		if verbose == 1:
			print type(bzTriple), bzTriple
		ptD_bezier_local = lst_3_pts[0]
		ptE_bezier_local = lst_3_pts[1]
		if verbose == 1:
			print "pts attache ds rep Bezier (B, C, D, E) : "
			print ptB_bezier_local
			print ptC_bezier_local
			print ptD_bezier_local
			print ptE_bezier_local
		ptLocation = self.posLineaire
		if verbose == 1:
			print "pt location ligament : ", ptLocation
		# Calcul des coordonnées des points d'attache dans le repère global 
		ptB_Global = fxMaths.convertTupleToArray( ptLocation ) + fxMaths.convertTupleToArray( ptB_bezier_local ) 
		ptC_Global = fxMaths.convertTupleToArray( ptLocation ) + fxMaths.convertTupleToArray( ptC_bezier_local ) 
		ptD_Global = fxMaths.convertTupleToArray( ptLocation ) + fxMaths.convertTupleToArray( ptD_bezier_local ) 
		ptE_Global = fxMaths.convertTupleToArray( ptLocation ) + fxMaths.convertTupleToArray( ptE_bezier_local ) 
		strNomObj_B, nNumSommet_B = self.getNomOsLePlusProche( ptB_Global )
		strNomObj_E, nNumSommet_E = self.getNomOsLePlusProche( ptE_Global )
		if verbose == 1:
			print "Point B proche du sommet num %d de %s\n"%( nNumSommet_B, strNomObj_B )
			print "Point E proche du sommet num %d de %s\n"%( nNumSommet_E, strNomObj_E )
			
		# Calcul des coordonnées du point d'attache dans le repère de l'objet strNomObj
		ptB_Global.resize( (4,) );# passage en coordonnées homogènes
		ptB_Global[3] = 1.0
		ptC_Global.resize( (4,) );# passage en coordonnées homogènes
		ptC_Global[3] = 1.0
		ptD_Global.resize( (4,) );# passage en coordonnées homogènes
		ptD_Global[3] = 1.0
		ptE_Global.resize( (4,) );# passage en coordonnées homogènes
		ptE_Global[3] = 1.0
		ptB_loc = self.calculerPositionPointGlobDansRepereObjet( ptB_Global, strNomObj_B )
		ptC_loc = self.calculerPositionPointGlobDansRepereObjet( ptC_Global, strNomObj_B )
		ptD_loc = self.calculerPositionPointGlobDansRepereObjet( ptD_Global, strNomObj_E )
		ptE_loc = self.calculerPositionPointGlobDansRepereObjet( ptE_Global, strNomObj_E )
		if verbose == 1:
			print "Pt attache dans le repere de l'os (B, C, D, E)"
			print ptB_loc
			print ptC_loc
			print ptD_loc
			print ptE_loc
		# Calcul des vecteurs de départ et d'arrivée de la Bézier
		vDepart = ptC_loc - ptB_loc
		vArrivee = ptD_loc - ptE_loc
		if verbose == 1:
			print "Vecteur depart"
			print vDepart
			print "Vecteur arrivee"
			print vArrivee
		
		# Mise à jour du fichier XML avec 
		# - ptB_loc, ptE_loc (extrémités de la Bezier)
		# - vDepart et vArrivee
		# - utilisation de strNomObj_B, strNomObj_E et self.strNomObjet pour trouver les noeuds XML
		if verbose == 1:
			print "\n   Mise a jour XML\n"
		fxXML = XML()
		fxXML.setNomFichierXML( self.strNomFichierXML )
		fxOs = Os()
		
		# Mise à jour de la position des éléments de la source de la Bézier
		strNomOs_B_court = fxOs.extraireNomOs( strNomObj_B )
		strNomAttache = self.strNomObjet + "_sur_" + strNomOs_B_court
		if verbose == 1:
			print "Recherche du noeud \"%s\" de \"%s\""%( strNomAttache, strNomOs_B_court )
		noeudAttache = fxXML.geNoeudAttache( strNomOs_B_court, strNomAttache )
		
		if ( noeudAttache != None ):
			print noeudAttache.toxml()
			#
			strVal = noeudAttache.getAttribute("x") 
			strM = "%.4f"%( ptB_loc[0] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("x", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("y") 
			strM = "%.4f"%( ptB_loc[1] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("y", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("y") 
			strM = "%.4f"%( ptB_loc[2] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("z", strM )
				bModif = True
			#
			# Patch du vecteur de départ
			strVal = noeudAttache.getAttribute("vx") 
			strM = "%.4f"%( vDepart[0] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("vx", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("vy") 
			strM = "%.4f"%( vDepart[1] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("vy", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("vz") 
			strM = "%.4f"%( vDepart[2] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("vz", strM )
				bModif = True
				
		# Mise à jour de la position des éléments de la destination de la Bézier
		strNomOs_E_court = fxOs.extraireNomOs( strNomObj_E )
		strNomAttache = self.strNomObjet + "_sur_" + strNomOs_E_court
		if verbose == 1:
			print "Recherche du noeud \"%s\" de \"%s\""%( strNomAttache, strNomOs_E_court )
		noeudAttache = fxXML.geNoeudAttache( strNomOs_E_court, strNomAttache )
		
		if ( noeudAttache != None ):
			print noeudAttache.toxml()
			#
			strVal = noeudAttache.getAttribute("x") 
			strM = "%.4f"%( ptE_loc[0] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("x", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("y") 
			strM = "%.4f"%( ptE_loc[1] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("y", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("y") 
			strM = "%.4f"%( ptE_loc[2] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("z", strM )
				bModif = True
			#
			# Patch du vecteur de départ
			strVal = noeudAttache.getAttribute("vx") 
			strM = "%.4f"%( vArrivee[0] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("vx", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("vy") 
			strM = "%.4f"%( vArrivee[1] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("vy", strM )
				bModif = True
			#
			strVal = noeudAttache.getAttribute("vz") 
			strM = "%.4f"%( vArrivee[2] )
			if ( strVal != strM ):
				noeudAttache.setAttribute("vz", strM )
				bModif = True
			
			# Mise à jour du fichier XML
			if bModif:
				print "enregistrement"
				fxXML.enregistrerFichier()
		
		fxXML.nettoyer()
		print "fin maj fichier xml"
