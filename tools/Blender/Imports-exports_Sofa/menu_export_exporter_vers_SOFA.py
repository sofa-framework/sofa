#!BPY
# -*- coding: iso8859-1 -*- 
"""
Name: 'Exporter la scène vers Sofa (.XML SOFA) ...'
Blender: 240
Group: 'Export'
Tooltip: 'Exporte la scène vers le fichier XML de Sofa (.XML)'
"""

__author__ = "Vincent Vansuyt"
__url__ = ("","")
__version__ = "1.0 2008/06/03"

__bpydoc__ = """\
Ce script met à jour un fichier XML natif de Sofa avec les positions de la scène.
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
	os.system('clear') # pour Linux

#sys.path.append("C:\\_Stage_VV\\Code\\scripts_python_Blender\\Export_vers_SOFA_v0_01")
#print sys.path
#import sp_Xfile


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
	strModule = strNomDossierInclude + os.sep + "main_CExport_vers_SOFA.py"
	if not os.path.exists( strModule ):
		strM = "Erreur|Attention : le fichier \"%s\" n'existe pas"%( strModule )
		Draw.PupMenu( strM )
	else:
		import main_CExport_vers_SOFA
		reload( main_CExport_vers_SOFA ) 
		from main_CExport_vers_SOFA import *
		
		print "Choix du fichier"
		fxExport = CExport_vers_SOFA( strNomDossierInclude + os.sep )
		print "Chemin scenes : \"%s\""%(fxExport.strChemin_scene_)
		if os.path.exists( fxExport.strChemin_scene_ ):
			os.chdir( fxExport.strChemin_scene_ )
		Window.FileSelector( fxExport.exporter, 'Mise à jour scène SOFA', '*.scn')
		#fxExport.exporter()
	
	# Rétablissement du chemin original pour ne pas "polluer" d'autres fonctions
	while strNomDossierInclude in sys.path:
		sys.path.remove( strNomDossierInclude )
print "----- fin du script -----"


