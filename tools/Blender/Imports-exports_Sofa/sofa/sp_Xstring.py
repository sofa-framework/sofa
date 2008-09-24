# -*- coding: iso8859-1 -*- 

class Xstring:
	#def __init__(self):
	#	print "constructeur"
	
	def afficher( self ):
		print "test"
		print self.str
	
	
	#
	# Renvoie True si la chaine strChaine commence par strDebut, False sinon
	#
	def bCommencePar( self, strChaine, strDebut ):
		#nLg = len( strDebut )
		#if len(strDebut) > len(strChaine):
		#	return False
		#strCut = strChaine[:nLg]
		#if strCut == strDebut:
		#	return True
		#else:
		#	return False
		return strChaine.startswith( strDebut )
	
	
	#
	# Renvoie True si la chaine strChaine fini par strFin, False sinon
	#
	def bFiniPar( self, strChaine, strFin ):
		return strChaine.endswidth( strFin )
	



