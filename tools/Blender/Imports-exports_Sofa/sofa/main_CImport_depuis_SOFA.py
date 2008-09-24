# -*- coding: iso8859-15 -*- 

from Blender import Draw, Window, Scene, NMesh, Mathutils, sys, Object, Text
from xml.dom import minidom, Node
import Blender
import BPyMesh
import BPyMessages
import os # Operating System (pas os de ossature)
import sys


import sp_Xstring
reload( sp_Xstring )
from sp_Xstring import *

import sp_Solide
reload( sp_Solide )
from sp_Solide import *

#import sp_Ligament
#reload( sp_Ligament )
#from sp_Ligament import *

import sp_XML
reload( sp_XML )
from sp_XML import *

import sp_Xfile
reload( sp_Xfile )
from sp_Xfile import *


class CImport_depuis_SOFA:
	
	verbose_ = False
	
	#
	# Initialise l'application
	#
	def __init__( self, strRepertoireRacineAppli ):
		self.strRepertoireRacineAppli_ = strRepertoireRacineAppli
		self.strNomFichierConfig_ = os.path.join( self.strRepertoireRacineAppli_ , "config.xml")
		# Recuperation des informations de configuration du fichier de configuration
		if ( not os.path.exists( self.strNomFichierConfig_ ) ):
			strM = "Erreur|Attention : le fichier \"%s\" est absent"%( self.strNomFichierConfig_ )
			Draw.PupMenu( strM )
			return
		fxXML = XML()
		fxXML.setNomFichierXML( self.strNomFichierConfig_ )
		
		# Recherche des noms de chemin configures
		noeudChemins = fxXML.getPremierNoeud()
		if noeudChemins == None:
			fxXML.nettoyer()
			return
		noeudChemins = noeudChemins.firstChild
		noeudChemins = fxXML.getNoeudSuivant( noeudChemins, "Path" )
		if ( noeudChemins != None):
			if noeudChemins.hasAttribute("strScenesPath"):
				self.strChemin_scenes_ = noeudChemins.getAttribute("strScenesPath")
			if noeudChemins.hasAttribute("strSofaPath"):
				self.strSofaPath_ = noeudChemins.getAttribute("strSofaPath")

		noeudFichiers = fxXML.getPremierNoeud().firstChild
		noeudFichiers = fxXML.getNoeudSuivant( noeudFichiers, "Files" )
		if ( noeudFichiers != None ):
			if noeudFichiers.hasAttribute("strNomCompletDerniereSceneImport"):
				self.strNomCompletFichierDerniereSceneImport_ = noeudFichiers.getAttribute("strNomCompletDerniereSceneImport")

		if( self.verbose_):
			print "------ XML Infos ------"
			print "Nom chemin scenes : \"%s\"" %(self.strChemin_scenes_ )
			print "Nom complet fichier derniere scene importee : \"%s\"" %(self.strNomCompletFichierDerniereSceneImport_ )
			print "-----------------------\n"
		fxXML.nettoyer()


	def importer( self, strNomFichierXML ):
		"""Importe la scène XML SOFA dans la scène Blender"""
		print "------ debut import -----"
		Window.WaitCursor(1)
		# Verification que le fichier XML à importer existe bien
		fxFichier = Xfile()
		if not fxFichier.bExiste( strNomFichierXML ):
			strM = "Attention|Le fichier \"%s\" n'existe pas."%( strNomFichierXML )
			Draw.PupMenu( strM )
			Window.WaitCursor(0)
			return
		
		#
		# Effacement de la scène avant l'import
		#
		self.effacer_la_scene()
		
		#
		# Parcours en profondeur du fichier XML et import des objets 
		#
		if( self.verbose_): print u"---- parcours de la scène à importer".encode( sys.stdout.encoding )
		# Initialisation des listes
		fxXML = XML()
		fxXML.setNomFichierXML( strNomFichierXML )
		strEncodage = "%s"%(fxXML.getNomCodage())
		if( self.verbose_): print u"Le fichier XML est au format \"%s\""%( strEncodage.encode( sys.stdout.encoding ) )
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
		
		# Parcours
		solide = Solide()
		nNiveauTrouve = -1
		strNomObjet = u"?"
		liste_des_nom_d_objet = []
		while ( nProfondeur >= 0 and listeIdxNoeuds[0] < listeCptNoeuds[ 0 ] ):
			# Affichage du chemin
			i = 0
			strPath = ""
			while i <= nProfondeur:
				noeud_tmp = listesListesNoeuds[ i ][ listeIdxNoeuds[i] ]
				if noeud_tmp.hasAttribute("name"):
					strNom = noeud_tmp.getAttribute("name")
				else:
					strNom = u"?"
				strPath += "\\" + strNom
				i += 1
			if( self.verbose_): print "* Path = \"%s\" -----( prof = %d ) ------- *"%(strPath, nProfondeur)
			
			noeudCourant = listesListesNoeuds[ nProfondeur ][ listeIdxNoeuds[nProfondeur] ]
			
			
			# Recherche d'un noeud "Object" dont la valeur de l'attribut "type" est "MechanicalObject"
			# ( mémorisation de la profondeur à laquelle on a trouvé cet objet )
			if nNiveauTrouve > 0 and nProfondeur <= nNiveauTrouve:
				nNiveauTrouve = -1
				if( self.verbose_): print "Ajout de l'objet '%s' dans la scene"%( solide.strNomObjet_ )
				solide.ajouter_dans_la_scene( Scene.GetCurrent(), self.strSofaPath_)#<<<<<<-----------------
				strNomObjet = "?"
				if( self.verbose_): print "\n" # désselection object

			if nNiveauTrouve == -1:
				listeNoeuds = fxXML.getListeNoeuds( noeudCourant.firstChild )
				for noeudParcours in listeNoeuds:
					if noeudParcours.hasAttribute("type"):
						strType = noeudParcours.getAttribute("type")
						if strType.lower() == "mechanicalobject":
							nNiveauTrouve = nProfondeur
							strNomObjet = strNom
							strNomObjet = self.calculer_nom_objet_sans_doublons( strNomObjet, liste_des_nom_d_objet)
							solide.setNomObjet( strNomObjet ) 
							solide.setCheminScene( self.strChemin_scenes_ )
							solide.initialiser()
							if( self.verbose_): print "Trouve objet : \"%s\", profondeur %d"%( strNomObjet, nProfondeur )
			
			# Traitement de la liste des noeuds pour l'objet strNomObjet
			if nNiveauTrouve > 0:
				listeNoeuds = fxXML.getListeNoeuds( noeudCourant.firstChild )
				for noeudParcours in listeNoeuds:
					solide.getInformations_depuis_noeud( noeudParcours)#<<<<<<-----------------
			
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
		
		# Ajout éventuel du dernier objet
		if strNomObjet != "?":
			solide.ajouter_dans_la_scene( Scene.GetCurrent(), self.strSofaPath_)#<<<<<<-----------------
		
		# Fermeture et nettoyage
		fxXML.nettoyer()
		Blender.Redraw()
		Window.WaitCursor(0)
		if( self.verbose_): print "------ fin import -----"
		return
		
		
	def calculer_nom_objet_sans_doublons( self, strNomObjet, liste_des_nom_d_objet ):
		"""Ajoute le nom d'objet dans la liste des objets en vérifiant l'absence de doublons sur les noms"""
		# Ajout de l'objet
		if strNomObjet in liste_des_nom_d_objet:
			# strNom est déjà dans la liste, calcul d'un autre nom
			nNum = 0
			strNomObjetNum = strNomObjet
			while strNomObjetNum in liste_des_nom_d_objet:
				nNum += 1
				strNum = "00000%d"%(nNum)
				nLg = len(strNum)
				strNum = strNum[nLg-3: nLg]
				strNomObjetNum = strNomObjet + "." + strNum
			strNomObjet = strNomObjetNum
		liste_des_nom_d_objet.append( strNomObjet )
		return strNomObjet
		
		
	def effacer_la_scene( self ):
		"""effacement de tous les maillages (on ne touche pas aux objets de type lampe, caméra, etc. """
		scene = Scene.GetCurrent() 
		lst_obj = scene.objects
		for obj in lst_obj:
			if obj.getType() == "Mesh":
				 scene.unlink( obj )
			# autres types d'objets à supprimer ?
	

