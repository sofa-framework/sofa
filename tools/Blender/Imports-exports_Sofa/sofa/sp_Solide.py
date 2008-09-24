# -*- coding: iso8859-1 -*-

import os
from xml.dom import minidom, Node
import Blender
from Blender import Object, Scene, Material
from math import pi
import string

import sp_Xstring
reload( sp_Xstring )
from sp_Xstring import *

import sp_XML
reload( sp_XML )
from sp_XML import *

import import_obj_local
reload( import_obj_local)
from import_obj_local import *

class Solide:
	
	def setParametresObjet( self, objet ):
		self.strNomObjet_ = objet.getName()
		self.posLineaire_ = objet.getLocation()
		self.dx_ = self.posLineaire_[0]
		self.dy_ = self.posLineaire_[1]
		self.dz_ = self.posLineaire_[2]
		self.posAngulaire_ = objet.getEuler()
		self.rx_ = self.posAngulaire_[0] * 180.0 / pi
		self.ry_ = self.posAngulaire_[1] * 180.0 / pi
		self.rz_ = self.posAngulaire_[2] * 180.0 / pi
		
		# Calcul des parametres min-max de l'objet (pour les "regular grid" par exemple)
		mesh = objet.getData(mesh=1)
		if len(mesh.verts) > 0:
			v = mesh.verts[0]
			self.xmin_ = v.co[0]
			self.ymin_ = v.co[1]
			self.zmin_ = v.co[2]
			self.xmax_ = self.xmin_
			self.ymax_ = self.ymin_
			self.zmax_ = self.zmin_
			for v in mesh.verts:
				if v.co[0] < self.xmin_:
					self.xmin_ = v.co[0]
				if v.co[0] > self.xmax_:
					self.xmax_ = v.co[0]
				#
				if v.co[1] < self.ymin_:
					self.ymin_ = v.co[1]
				if v.co[1] > self.ymax_:
					self.ymax_ = v.co[1]
				#
				if v.co[2] < self.zmin_:
					self.zmin_ = v.co[2]
				if v.co[2] > self.zmax_:
					self.zmax_ = v.co[2]
		# Récupération de la couleur
		self.nbMateriaux_ = len( mesh.materials )
		self.rgbCol_ = [0.8, 0.8, 0.8]
		if self.nbMateriaux_ > 0:
			mat = mesh.materials[0]
			print type(mat )
			self.rgbCol_ = mat.rgbCol
			
	
	
	
	def setParametresXML( self, strNomFichierXML ):
		self.strNomFichierXML_ = strNomFichierXML	
	
	
	
	def afficher( self ):
		print "\n"
		print "-------------------------------------------"
		print "Nom objet : \"%s\""%( self.strNomObjet_ )
		print "dx = %.3f, dy = %.3f, dz=%.3f"%(self.dx_, self.dy_, self.dz_)
		print "rx = %.3f, ry = %.3f, rz=%.3f"%(self.rx_, self.ry_, self.rz_)
		print "xmin = %.3f, ymin = %.3f, zmin = %.3f"%(self.xmin_, self.ymin_, self.zmin_)
		print "xmax = %.3f, ymax = %.3f, zmax = %.3f"%(self.xmax_, self.ymax_, self.zmax_)
		print "Nb materiaux = %d"%(self.nbMateriaux_)
		print "Couleur R %.3f, G %.3f, B %.3f"%( self.rgbCol_[0], self.rgbCol_[1], self.rgbCol_[2])
		print "-------------------------------------------"
		print "\n"
	
	
	
	def mettre_fichier_XML_a_jour( self ):
		"""Parcours le fichier XML en profondeur"""
		print "Debut parcours pour l'objet \"%s\""%( self.strNomObjet_ )
		
		# Initialisation des listes
		fxXML = XML()
		fxXML.setNomFichierXML( self.strNomFichierXML_ )
		noeud = fxXML.getPremierNoeud()
		
		listesListesNoeuds = [[noeud, noeud]]
		listeCptNoeuds = [0]
		listeIdxNoeuds = [0]
		
		# Initialisation des valeurs du niveau zéro
		nProfondeur = 0
		listeNoeuds = fxXML.getListeNoeuds( noeud )
		listeNoeuds = fxXML.getListeAvecUniquementNoeudsParents( listeNoeuds )
		listesListesNoeuds[ nProfondeur ] = listeNoeuds
		# self.afficherListeNoeuds( listesListesNoeuds[ nProfondeur ] )
		listeCptNoeuds[ nProfondeur ] = len( listesListesNoeuds[ nProfondeur ] )
		listeIdxNoeuds[ nProfondeur ] = 0
		
		nNiveauTrouve = -1
		while ( nProfondeur >= 0 and listeIdxNoeuds[0] < listeCptNoeuds[ 0 ] ):
			# Affichage du chemin
			i = 0
			strPath = ""
			while i <= nProfondeur:
				noeud_tmp = listesListesNoeuds[ i ][ listeIdxNoeuds[i] ]
				if noeud_tmp.hasAttribute("name"):
					strNom = noeud_tmp.getAttribute("name")
				else:
					strNom = "?"
				strPath += "\\" + strNom
				i += 1
			print "* Path = \"%s\" ------------ *"%(strPath)
			
			noeudCourant = listesListesNoeuds[ nProfondeur ][ listeIdxNoeuds[nProfondeur] ]
			
			# Détection des noeuds à traiter
			if self.strNomObjet_ == strNom:
				nNiveauTrouve = nProfondeur
				
			if self.strNomObjet_ != strNom and nProfondeur <= nNiveauTrouve:
				nNiveauTrouve = -1 # fin des patchs
			
			# Traiter noeud 
			if nNiveauTrouve > 0:
				print "**>> Patch !!"
				listeNoeuds = fxXML.getListeNoeuds( noeudCourant.firstChild )
				fxXML.afficherListeNoeuds( listeNoeuds )
				for noeud_a_patcher in listeNoeuds:
					self.patcherNoeud( noeud_a_patcher )
			
			# Profondeur suivante ?
			if ( noeudCourant.firstChild != None ) : # profondeur suivante
				nProfondeur += 1
				if ( nProfondeur >= len( listesListesNoeuds ) ):
					# ajout d'un element dans la liste
					listesListesNoeuds += [[noeud]]
					listeCptNoeuds += [ 0 ]
					listeIdxNoeuds += [ 0 ]
				# Mise à jour des éléments du niveau suivant
				listeNoeuds = fxXML.getListeNoeuds( noeudCourant.firstChild )
				listeNoeuds = fxXML.getListeAvecUniquementNoeudsParents( listeNoeuds )
				listesListesNoeuds[ nProfondeur ] = listeNoeuds
				listeCptNoeuds[ nProfondeur ] = len( listeNoeuds )
				listeIdxNoeuds[ nProfondeur ] = 0
				
			elif ( listeIdxNoeuds[ nProfondeur ] < listeCptNoeuds[ nProfondeur ] ):
				# Passage au dossier suivant dans la meme profondeur
				listeIdxNoeuds[ nProfondeur ] += 1
			
			# Remontage d'un niveau si necessaire
			while ( nProfondeur > 0 and listeIdxNoeuds[ nProfondeur ] >= listeCptNoeuds[ nProfondeur ]):
				nProfondeur-= 1
				listeIdxNoeuds[ nProfondeur ] += 1
		
		# fin
		#print "enregistrement"
		#print fxXML.getPremierNoeud().toxml()
		fxXML.enregistrerFichier()
		fxXML.nettoyer()
		print "fin maj fichier xml"
	
	
	def patcherNoeud( self, noeud ):
		"""Patche le noeud recu avec les valeurs presentent dans le solide"""
		# Taille de grille
		if noeud.hasAttribute("nx") or noeud.hasAttribute("ny") or noeud.hasAttribute("nz"):
			print noeud.toxml()
		# Position linéaire
		if noeud.hasAttribute("dx") or noeud.hasAttribute("dy") or noeud.hasAttribute("dz"):
			# X
			if noeud.hasAttribute("dx"):
				noeud.setAttribute("dx", "%.4f"%(self.dx_))
			# Y
			if noeud.hasAttribute("dy"):
				noeud.setAttribute("dy", "%.4f"%(self.dy_))
			# Z
			if noeud.hasAttribute("dz"):
				noeud.setAttribute("dz", "%.4f"%(self.dz_))
		
		# Position angulaire
		if noeud.hasAttribute("rx") or noeud.hasAttribute("ry") or noeud.hasAttribute("rz"):
			# rx
			if noeud.hasAttribute("rx"):
				noeud.setAttribute("rx", "%.4f"%(self.rx_))
			# ry
			if noeud.hasAttribute("ry"):
				noeud.setAttribute("ry", "%.4f"%(self.ry_))
			# rz
			if noeud.hasAttribute("rz"):
				noeud.setAttribute("rz", "%.4f"%(self.rz_))
			
		# Couleur
		if noeud.hasAttribute("color"):
			noeud.setAttribute("color", "%.3f %.3f %.3f"%(self.rgbCol_[0], self.rgbCol_[1], self.rgbCol_[2]))
		
		# Nom de fichier obj
		if noeud.hasAttribute("fileMesh"):
			print noeud.toxml()



	def convertirChaineCouleurEnChaineRGB( self, strCouleur ):
		"""Convertir une chaine de couleur en trois composantes RGB de 0 à 1 concatennees"""
		strCouleur = strCouleur.lower()
		if " " in strCouleur:# la couleur est une séquence de r g b ("0.5 0.5 0.1" par exemple)
			return strCouleur
		# La couleur est un nom
		strRgb = "0.8 0.8 0.8" # gris par défaut (r g b)
		if (strCouleur == "white") :
			strRgb = "1.0 1.0 1.0"
		elif (strCouleur == "black") :
			strRgb = "0.0 0.0 0.0" 
		elif (strCouleur == "red") :
			strRgb = "1.0 0.0 0.0" 
		elif (strCouleur == "green") : 
			strRgb = "0.0 1.0 0.0" 
		elif (strCouleur == "blue") : 
			strRgb = "0.0 0.0 1.0"
		elif (strCouleur == "cyan") :  
			strRgb = "0.0 1.0 1.0" 
		elif (strCouleur == "magenta") : 
			strRgb = "1.0 0.0 1.0" 
		elif (strCouleur == "yellow") : 
			strRgb = "1.0 1.0 0.0"  
		elif (strCouleur == "gray") : 
			strRgb = "0.5 0.5 0.5" 
		return strRgb



	def initialiser( self ):
		# Initialisation des éléments par défaut
		self.posLineaire_ = [0.0, 0.0, 0.0]
		self.posAngulaire_ = [0.0, 0.0, 0.0]
		self.rgbCol_ = [0.8, 0.8, 0.8]
		self.strFileName_ = ""
	
	
	
	def setNomObjet( self, strNom ):
		nLgMaxBlender = 21
		nLg = len( strNom )
		#if ( nLg > nLgMaxBlender ):
		#	strNomCourt = strNom[0:nLgMaxBlender-1]
		#	strM =  u"Attention : le nom d'objet \"" + strNom + "\" a été tronqué en \"" + strNomCourt + "\"."
		#	print strM.encode(sys.stdout.encoding,'replace')
		#	strM = u"(Blender est limite à %d caractères)"%(nLgMaxBlender)
		#	print strM.encode(sys.stdout.encoding,'replace')
		#	#strNom = strNomCourt
		self.strNomObjet_ = strNom
	
	
	def setCheminScene( self, strCheminScene ):
		self.strCheminScene_ = strCheminScene



	def getInformations_depuis_noeud( self, noeud ):
		if not noeud.hasAttribute("type"):
			return 
		if noeud.hasAttribute("dx"):
			self.posLineaire_[0] = eval( noeud.getAttribute("dx") )
		if noeud.hasAttribute("dy"):
			self.posLineaire_[1] = eval( noeud.getAttribute("dy") )
		if noeud.hasAttribute("dz"):
			self.posLineaire_[2] = eval( noeud.getAttribute("dz") )
		#
		if noeud.hasAttribute("rx"):
			self.posAngulaire_[0] = eval( noeud.getAttribute("rx") ) * pi / 180.0
		if noeud.hasAttribute("ry"):
			self.posAngulaire_[1] = eval( noeud.getAttribute("ry") ) * pi / 180.0
		if noeud.hasAttribute("rz"):
			self.posAngulaire_[2] = eval( noeud.getAttribute("rz") ) * pi / 180.0
		if noeud.hasAttribute("color"):
			strColor = noeud.getAttribute("color")
			strRGB = self.convertirChaineCouleurEnChaineRGB( strColor)
			lstStrColor = strRGB.split(" ")
			self.rgbCol_[0] = eval(lstStrColor[0])
			self.rgbCol_[1] = eval(lstStrColor[1])
			self.rgbCol_[2] = eval(lstStrColor[2])
			
		if (self.strFileName_ == ""):
			if noeud.hasAttribute("filename"):
				self.strFileName_ = noeud.getAttribute("filename")
			if noeud.hasAttribute("fileMesh"):
				self.strFileName_ = noeud.getAttribute("fileMesh")
			if not self.strFileName_.endswith(".obj"):
				self.strFileName_ = ""



	def ajouter_dans_la_scene( self, scene, strSofaPath):
		if self.strFileName_ == "":
			print "sp_Solide.py::ajouter_dans_la_scene(): Nom de fichier vide"
		
		# Recherche des fichiers dans le repertoire courant ou les repertoires Sofa
		sofaPath = os.path.join( strSofaPath, "share")
		
		# On vérifie si le fichier est dans le dossier de Sofa
		if os.path.exists( os.path.join( sofaPath , self.strFileName_)):
			path_ = sofaPath
		# On vérifie dans le dossier courant qui prévaut sur le dossier Sofa
		if os.path.exists( os.path.join( self.strCheminScene_, self.strFileName_)):
			path_ = self.strCheminScene_
		if not os.path.exists( os.path.join( path_, self.strFileName_)):
			print "sp_Solide.py::ajouter_dans_la_scene(): Attention : le fichier \"%s\" n'existe pas"%( self.strFileName_ )
		else:
			os.chdir( path_)
			object = load_obj( self.strFileName_)[0]
			object.setName( self.strNomObjet_ )
			object.setLocation( self.posLineaire_[0], self.posLineaire_[1], self.posLineaire_[2] )
			object.setEuler( self.posAngulaire_ ) # ( en radians )
			
			strNomMat = u"mat" + self.strNomObjet_
			mat = Material.New( strNomMat.encode('ISO8859-1') ) 
			mat.setRGBCol( self.rgbCol_ )
			object.getData( False, True).materials += [mat]

