# -*- coding: iso8859-15 -*- 

from Blender import Draw, Window, Scene, NMesh, Mathutils, sys, Object, Scene, Text
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

import sp_CExport_fichiers3D
reload( sp_CExport_fichiers3D )
from sp_CExport_fichiers3D import *

import sp_XML
reload( sp_XML )
from sp_XML import *

import sp_Xfile
reload( sp_Xfile )
from sp_Xfile import *


class CExport_vers_SOFA:
	
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
			if noeudChemins.hasAttribute("strSofaPath"):
				self.strSofaPath_ = noeudChemins.getAttribute("strSofaPath")
				# Temporaire... Virer les deux suivants?
				self.strChemin_modeles_visuels_ = os.path.join( self.strSofaPath_, "share")
				self.strChemin_modeles_collision_ = os.path.join( self.strSofaPath_, "share")
			#
			if noeudChemins.hasAttribute("strScenesPath"):
				self.strChemin_scene_ = noeudChemins.getAttribute("strScenesPath")
		
		# Recherche des noms de fichiers configurés
		noeudFichiers = fxXML.getPremierNoeud()
		noeudFichiers = fxXML.getPremierNoeud()
		if noeudFichiers == None:
			fxXML.nettoyer()
			return
		noeudFichiers = noeudFichiers.firstChild
		noeudFichiers = fxXML.getNoeudSuivant( noeudFichiers, "Files" )
		if ( noeudFichiers != None ):
			if noeudFichiers.hasAttribute("strNomCompletExecutableSofa"):
				self.strNomCompletExecutableSofa_ = noeudFichiers.getAttribute("strNomCompletExecutableSofa")
		print "Nom fichier executable sofa  : \"%s\""%(self.strNomCompletExecutableSofa_ )
		print "Nom chemin modeles visuels   : \"%s\"" %(self.strChemin_modeles_visuels_ )
		print "Nom chemin modeles collision : \"%s\"" %(self.strChemin_modeles_collision_ )
		print "Nom chemin scene             : \"%s\""%(self.strChemin_scene_ )
		fxXML.nettoyer()
	
	
	#
	# Exporte la scène dans le fichier XML
	#
	def exporter( self, strNomFichierXML ):
		print "----- debut export ----"
		# Récupération du fichier XML à patcher et vérification de son existence
		if not os.path.isfile( strNomFichierXML ):
			strM = "Attention|Le fichier XML n'existe pas (%s)" %( strNomFichierXML )
			Draw.PupMenu( strM )
			print strM
			return
			
		# Récupération de réglages complémentaires
		EXPORT_FICHIERS_OBJ_VISU = Draw.Create(0)
		EXPORT_FICHIERS_OBJ_COLLISION = Draw.Create(0)
		
		
		# Récupération des options de l'utilisateur
		pup_block = []
		pup_block.append('Choix des informations à exporter :');
		pup_block.append(('Maillages visualisation', EXPORT_FICHIERS_OBJ_VISU, 'Enregistrer les maillages dans le dossier de visualisation'));
		pup_block.append(('Maillages collision', EXPORT_FICHIERS_OBJ_COLLISION, 'Enregistrer les maillages dans le dossier de collision'));
		if not Draw.PupBlock('Choix des informations à exporter', pup_block):
			return
		
		# Parcours de la scène à la recherche des objets à traiter
		Window.WaitCursor(1)
		scene = Scene.getCurrent()
		lst_obj = scene.objects 
		
		fxStr = Xstring()
		fxFichier = Xfile()
		fxExportFichiers3D = CExport_fichiers3D()
		
		# - parcours des objets de la scène et mise à jour du fichier XML
		solide = Solide()
		solide.setParametresXML( strNomFichierXML  )
		for obj in lst_obj:
			#obj = lst_obj[0]
			if obj.getType() == "Mesh":
				strNom = obj.getName()
				print strNom
				solide.setParametresObjet( obj )
				solide.afficher()
				
				strNomFichierObj_visuel = \
							fxFichier.concatenerChemins( self.strChemin_modeles_visuels_ , strNom + ".obj")
				strNomFichierObj_collision = \
						fxFichier.concatenerChemins( self.strChemin_modeles_collision_ , strNom + ".obj")
				if EXPORT_FICHIERS_OBJ_VISU.val == 1:
					fxExportFichiers3D.writeObj( obj, strNomFichierObj_visuel )
				else:
					print "Pas d'enregistrement du fichier \"%s\""%( strNomFichierObj_visuel )
				if EXPORT_FICHIERS_OBJ_COLLISION.val == 1:
					fxExportFichiers3D.writeObj( obj, strNomFichierObj_collision )
				else:
					print "Pas d'enregistrement du fichier \"%s\""%( strNomFichierObj_collision )
				
				# - Patch du fichier XML (positions, couleurs, min et max pour les "regular grid")
				solide.mettre_fichier_XML_a_jour() # ---->>>>>>>>
		
		Window.WaitCursor(0)
		return
		
		
