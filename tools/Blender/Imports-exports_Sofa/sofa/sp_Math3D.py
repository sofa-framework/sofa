# -*- coding: iso8859-1 -*- 

from math import *
from numpy import *
from scipy.linalg import *

class Math3D:
	
	#
	# Passe le tableau en coordonnées homogènes
	#
	def passerEnCoordonneesHomogenes( self, v_in ):
		#v_out = v_in
		if len( v_in ) == 3:
			v_in.resize( (4,) )# passage en coordonnées homogènes
			v_in[3] = 1.0
		return v_in
	
	
	#
	# Convertit un "Tuple" en array
	#
	def convertTupleToArray( self, T ):
		nLg = len( T )
		A = zeros( ( nLg ), dtype=double ) 
		i = 0
		while (i < nLg ):
			#print T[i]
			A[i] = T[i]
			i = i + 1
		return A
	
	
	#
	# Calcul un quaternion à partir des angles d'Euler a0 a1 a2 (en radians), autour des axes x, y et z
	# ( d'après l'article de Wikipédia)
	def calcQuaternionWiki( self, a0, a1, a2 ):
		# calcul des 4 valeurs du quaternion
		Q = array([0.0, 0.0, 0.0, 0.0]);
		Q[0] = cos(a0/2)*cos(a1/2)*cos(a2/2) + sin(a0/2)*sin(a1/2)*sin(a2/2);
		Q[1] = sin(a0/2)*cos(a1/2)*cos(a2/2) - cos(a0/2)*sin(a1/2)*sin(a2/2);
		Q[2] = cos(a0/2)*sin(a1/2)*cos(a2/2) + sin(a0/2)*cos(a1/2)*sin(a2/2);
		Q[3] = cos(a0/2)*cos(a1/2)*sin(a2/2) - sin(a0/2)*sin(a1/2)*cos(a2/2); 
		return Q;
	
	
	#
	# Renvoie une matrice de rotation (4x4) à partir d'un Quaternion
	#
	def quat_to_matrix( self, Q ):
		# w,x,y,z;
		w = Q[0];
		x = Q[1];
		y = Q[2];
		z = Q[3];
		xx = 2 * x * x
		xy = 2 * x * y
		xz = 2 * x * z
		yy = 2 * y * y
		yz = 2 * y * z
		zz = 2 * z * z
		wx = 2 * w * x
		wy = 2 * w * y
		wz = 2 * w * z;
		
		M = zeros( ( 4, 4 ), dtype=double ) 
		M[ 0, 0 ] = 1 - yy - zz
		M[ 0, 1 ] = xy - wz
		M[ 0, 2 ] = xz + wy
		
		M[ 1, 0 ] = xy + wz 
		M[ 1, 1 ] = 1 - xx - zz
		M[ 1, 2 ] = yz - wx
		
		M[ 2, 0 ] = xz - wy
		M[ 2, 1 ] = yz + wx
		M[ 2, 2 ] = 1 - xx - yy
		
		M[ 3, 3 ] = 1;
		return M;
	
	
	#
	# Calcule une matrice de rotation à partir d'un vecteur 
	#  translation
	#
	def calcMatriceTranslation( self, vT ):
		M = zeros( ( 4, 4 ), dtype=double ) 
		M[ 0, 0 ] = 1.0
		M[ 1, 1 ] = 1.0
		M[ 2, 2 ] = 1.0
		M[ 3, 3 ] = 1.0
		M[ 0, 3 ] = vT[ 0 ]
		M[ 1, 3 ] = vT[ 1 ]
		M[ 2, 3 ] = vT[ 2 ]
		return M;
	
	
	
	
	
