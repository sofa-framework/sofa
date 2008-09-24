#!BPY
# -*- coding: iso8859-1 -*- 
"""
Name: 'Importer une scène de Sofa (.XML SOFA) ...'
Blender: 240
Group: 'Import'
Tooltip: 'Importe une scène XML de Sofa vers Blender'
"""

__author__ = "Vincent Vansuyt"
__url__ = ("","")
__version__ = "0.02 2008/06/04"

# Historique
#
# v0.01 : Auteur : Vincent Vansuyt
#       : - première version
#       : 
#
# v0.02 : Auteur : Vincent Vansuyt
#       : - gestion des noms d'objets accentués
#       : - amélioration de la gestion des noms de fichiers obj au niveau des chemins
#       : 
#

__bpydoc__ = """\
Ce script importe une scène XML Sofa dans Blender
"""

from Blender import Draw, Window, Scene, NMesh, Mathutils, sys, Object, Scene, Text
import Blender
import BPyMesh
import BPyMessages
import os # Operating System (pas os de ossature)
import sys
import platform
#print "%s"%(sys.path)

# Effacement du texte de la console
if ( platform.system() == "Windows" ):
	os.system('cls') # pour Windows
else:
	os.system('clear') # pour Linux ou Mac


print "----- debut du script -----"
# Nom du dossier des scripts utilisateurs
strNomDossierScriptsUser = Blender.Get('uscriptsdir')
strNomDossierInclude = os.path.join( strNomDossierScriptsUser , "sofa" )# <<< CONFIGURER ICI <<<<
print strNomDossierInclude
if not os.path.exists( strNomDossierInclude ):
	strM = "Erreur|Attention : le dossier \"%s\" n'existe pas"%( strNomDossierInclude )
	Draw.PupMenu( strM )
else:
	if not strNomDossierInclude in sys.path:
		sys.path.append( strNomDossierInclude )
	#sys.path.sort
	#print sys.path
	# Blender.Get('filename') # fichier par défaut
	strModule = os.path.join( strNomDossierInclude, "main_CImport_depuis_SOFA.py")
	if not os.path.exists( strModule ):
		strM = "Erreur|Attention : le fichier \"%s\" n'existe pas"%( strModule )
		Draw.PupMenu( strM )
	else:
		import main_CImport_depuis_SOFA
		reload( main_CImport_depuis_SOFA ) 
		from main_CImport_depuis_SOFA import *

		fxImport = CImport_depuis_SOFA( strNomDossierInclude)
		if os.path.exists( fxImport.strChemin_scenes_ ):
			os.chdir( fxImport.strChemin_scenes_ )
		Blender.Window.FileSelector( fxImport.importer, 'Import scène SOFA', '*.scn')
		
	# Rétablissement du chemin original pour ne pas "polluer" d'autres fonctions
	while strNomDossierInclude in sys.path:
		sys.path.remove( strNomDossierInclude )
print "----- fin du script -----"

