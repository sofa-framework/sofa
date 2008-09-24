# -*- coding: iso8859-1 -*- 

import os.path
from xml.dom import minidom, Node
#import xml.dom.ext
#import pprint

import sp_Xstring
reload( sp_Xstring )
from sp_Xstring import *

class XML:
	dicoTypesNoeuds = { 'element':1, 'commentaire':8 }
	#ELEMENT_NODE = 1
	#ATTRIBUTE_NODE = 2
	#TEXT_NODE = 3
	#CDATA_SECTION_NODE = 4
	#ENTITY_REFERENCE_NODE = 5
	#ENTITY_NODE = 6
	#PROCESSING_INSTRUCTION_NODE = 7
	#COMMENT_NODE = 8
	#DOCUMENT_NODE = 9
	#DOCUMENT_TYPE_NODE = 10
	#DOCUMENT_FRAGMENT_NODE = 11
	#NOTATION_NODE = 12
	
	def setNomFichierXML( self, strNomFichierXML ):
		self.strNomFichierXML = strNomFichierXML
		if not os.path.exists( strNomFichierXML ):
			print "XML::setNomFichierXML : attention ! Le fichier \"%s\" n'existe pas."%( strNomFichierXML )
		else:
			self.xmldoc = minidom.parse( strNomFichierXML )
	
	
	def get_xmldoc( self ):
		return self.xmldoc
	
	
	def getNomCodage( self ):
		"""Renvoie le nom du codage utilisé pour le fichier XLM ouvert"""
		return self.xmldoc.encoding
	
	
	def getPremierNoeud( self ):
		noeud = self.xmldoc.firstChild
		noeud = self.getNoeudSuivantSansCommentaire( noeud )
		return noeud
	
	
	def getNoeudSuivantSansCommentaire( self, noeud ):
		"Passe les noeuds commentaires commentaires et renvoie le noeud suivant"
		#nNumTypeCommentaire = self.dicoTypesNoeuds['commentaire']
		#print "Num type commentaire = %d"%(nNumTypeCommentaire)
		while ( noeud != None and \
				  noeud.nodeType == Node.COMMENT_NODE or \
				  noeud.nodeType == Node.TEXT_NODE and
				  noeud != noeud.lastChild ):
			noeud = noeud.nextSibling
		return noeud
	
	
	def getNoeudSuivant( self, noeud, strNom ):
		#print "Recherche du noeud %s"%(strNom)
		"Renvoie le prochain noeud frère dont on donne le nom"
		#print "Noeud \"%s\""%(noeud.nodeName)
		noeud = noeud.nextSibling
		while ( noeud != None and noeud.nodeName != strNom ):
			noeud = noeud.nextSibling
			#print "Noeud \"%s\""%(noeud.nodeName)
		return noeud
	
	
	#
	# Recherche un frère de noeud nommé strNomNoeudAttendu qui a un attribut strNomAttributAttendu dont 
	#  la valeur est strValeurAttribut
	#
	def getNoeudSuivantAvecAttribut( self, noeud, strNomNoeudAttendu, strNomAttributAttendu, strValeurAttribut ):
		bTrouve = False
		while ( noeud != None and not bTrouve ):
			if ( noeud.nodeName == strNomNoeudAttendu ):
				if ( noeud.hasAttribute( strNomAttributAttendu ) ):
					if ( noeud.getAttribute( strNomAttributAttendu ) == strValeurAttribut ) :
						bTrouve = True
			if not bTrouve:
				noeud = noeud.nextSibling
			#print "Noeud \"%s\""%(noeud.nodeName)
		return noeud
	
	
	def getNoeudOs( self, strNomOs ):
		noeud = self.xmldoc.firstChild
		noeud = self.getNoeudSuivant( noeud, "scene" )
		noeud = noeud.firstChild
		noeud = self.getNoeudSuivant( noeud, "elements_rigides" )
		noeud = noeud.firstChild
		noeud = self.getNoeudSuivant( noeud, "os" )
		#print "Recherche du noeud \""+ strNomOs + "\""
		while ( noeud != None and noeud.getAttribute('nom') != strNomOs ):
			noeud = self.getNoeudSuivant( noeud, "os" )
		#print noeud.nodeName 
		#print noeud.toxml()
		return noeud
	
	
	def geNoeudAttache( self, strNomOs, strNomAttache ):
		# Recherche du noeud du nom de l'os, par exemple "<os nom="tibia" fixe="1" masse="3.0">"
		noeudAttache = None
		noeudOs = self.getNoeudOs( strNomOs )
		if ( noeudOs != None ):
			noeudAttaches = noeudOs.firstChild
			noeudAttaches = self.getNoeudSuivant( noeudAttaches, "attaches" )
			if ( noeudAttaches != None ):
				noeudAttache = noeudAttaches.firstChild
				noeudAttache = self.getNoeudSuivantSansCommentaire( noeudAttache )
				while ( noeudAttache != None and noeudAttache.getAttribute('nom') != strNomAttache ):
					noeudAttache = self.getNoeudSuivant( noeudAttache, "attache" )
		return noeudAttache
	
	
	def afficher( self ):
		print "Nom fichier XML : \"%s\""%( self.strNomFichierXML )
		print self.xmldoc.toxml()
	
	
	def nettoyer( self ):
		self.xmldoc.unlink()
	
	
	def enregistrerFichier( self ):
		#strChemin = os.path.dirname( self.strNomFichierXML )
		#strNomTmp = strChemin + "\\out.xml" 
		fichier = open(self.strNomFichierXML, "w")
		self.xmldoc.writexml( fichier, "  ", "", "", "ISO-8859-1" )
		fichier.close()
		#print "fichier XML enregistre dans le fichier %s"%(self.strNomFichierXML)
	
	
	#
	# Renvoie la liste de noeuds
	#
	def getListeNoeuds( self, noeud_in ):
		# retour au debut de la liste
		noeud = noeud_in
		while ( noeud != None ):
			noeudSov = noeud
			noeud = noeud.previousSibling
		noeud = noeudSov
		# constitution de la liste
		liste = []
		nbNoeuds = 0
		while ( noeud != None ):
			if (noeud.nodeType == Node.ELEMENT_NODE):
				nbNoeuds = nbNoeuds + 1
				if (len( liste ) == 0):
					liste = noeud
					print noeud.nodeName
				else:
					liste = [liste, noeud]
					print "noeud %s, type %s"%(noeud.nodeName, noeud.nodeType)
			noeud = noeud.nextSibling
		print "Nb noeuds %d"%(nbNoeuds )
		return liste
	
	
	#
	# Afficher liste noeuds
	# 
	def afficherListeNoeuds( self, liste_noeuds ):
		nbNoeuds = len( liste_noeuds )
		print "nb noeuds = %d"%(nbNoeuds)
		i = 0
		while i < nbNoeuds:
			print "+ num %d, "%( i )
			i = i + 1
		
	
	
	def getListeNoeuds( self, noeud_in ):
		"""Renvoie la liste des noeuds du même niveau que noeud_in"""        
		liste = []
		if noeud_in == None:
			return liste
		noeudParent = noeud_in.parentNode
		if noeudParent == None:
			return liste
		liste = noeudParent.childNodes
		# Suppression des noeuds texte de la liste
		liste_retour = []
		for noeud in liste:
			if noeud.nodeType == Node.ELEMENT_NODE:
				liste_retour.append( noeud )
		return liste_retour
	
	
	def getListeAvecUniquementNoeudsParents(self, listeNoeuds ): 
		# Supprime les noeuds qui n'ont pas d'enfants
		liste_retour = []
		bSuppr = True
		for noeud in listeNoeuds:
			if noeud.firstChild != None:
				liste_retour.append( noeud )
		return liste_retour
		
		
	def afficherListeNoeuds( self, liste_noeuds ):
		"""Affiche la liste des noeuds reçus en parametre"""
		nbNoeuds = len( liste_noeuds )
		print "nb noeuds = %d"%(nbNoeuds)
		i = 0
		for noeud in liste_noeuds:
			if noeud.hasAttribute("type"):
				strType = noeud.getAttribute("type")
			else:
				strType = "?"                
			if noeud.hasAttribute("name"):
				strNom = noeud.getAttribute("name")
			else:
				strNom = "?"              
			print "+ num %d, %s name = \"%s\", type = \"%s\""%( i, liste_noeuds[i].nodeName, strNom, strType )
		


