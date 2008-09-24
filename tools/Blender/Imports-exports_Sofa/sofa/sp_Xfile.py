# -*- coding: iso8859-1 -*- 

import os 
import sys

class Xfile:
	#def __init__(self):
	#	print "constructeur"
	
	#
	# Renvoie True si fichier (ou chemin) reçu existe, False sinon
	#
	def bExiste( self, strFichier ):
		return os.path.exists( strFichier ) 
	
	
	#
	# Copie le fichier source vers le fichier destination
	#
	def copierFichier( self, strNomFichierSource, strNomFichierDestination ):
		file = open( strNomFichierSource, 'rb')
		data = file.read()
		file.close()
	
		file = open( strNomFichierDestination, 'wb' )
		file.write( data )
		file.close()
	
	
	#
	# Renvoie la concaténation de strPathDebut + strPathFin en tenant compte du caractère de 
	#  séparation de chemins
	#
	def concatenerChemins( self, strPathDebut, strPathFin ):
		return os.path.join( strPathDebut, strPathFin )
	
	
	#
	# Renvoie le caractère de séparation des dossiers "\\" sous Windows "/" sous Linux
	#
	def getCaractereSeparateurDeChemins( self ):
		return os.sep
	
	
	#
	# Verifie que le fichier se termine bien par strNomExtension
	# Renvoie le nom du fichier éventuellement corrigé
	#
	def verifierExtensionFichier( self, strNomFichier, strNomExtension ):
		if not strNomExtension.beginswith('.'):
			strNomExtension = "." + strNomExtension;
		if not strNomFichier.lower().endswith( strNomExtension ): 
			strNomFichier += '.obj'
	

