#!/usr/bin/env python
# -*- coding: utf-8 -*-
from math import *

""" *****************************************************************************
*                               SOFA :: extlibs                                  *
*                                                                                *
* Authors: Dcko Ali Hamadi													     *
*                                                                                *
* Contact information: dickoah@gmail.com		                                 *
 ******************************************************************************"""
#*********************************************************************************************
#
# Angles d'Euler | Quaternion
#
#*********************************************************************************************
"""---------------------------------------------------------------
| norm_Quat
| Description : Retourne la norme d'un quaternion
   ---------------------------------------------------------------"""
def norm_Quat( q ):
	q0 = q[3]
	q1 = q[0]
	q2 = q[1]
	q3 = q[2]
	return sqrt( q0*q0 + q1*q1 + q2*q2 + q3*q3 ) 

"""---------------------------------------------------------------
| normalize_Quat
| Description : Normalise un quaternions
 ---------------------------------------------------------------"""
def normalize_Quat( q ):
	norm = norm_Quat(q)
	if norm > 1E-8:
		for i in range(0,4):
			q[i] = q[i]/norm 
	return q

"""---------------------------------------------------------------
| Quat_to_n_theta
| Description : Transforme un quaternion en n et theta
 ---------------------------------------------------------------"""
def quat_to_n_theta( q, n=[0., 0., 0.], theta=0.0 ) :
	theta = 2.0*acos(q[3])
	sinus = sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2]) 
	if(sinus > 1E-10):
		n[0] = q[0]/(sinus) 
		n[1] = q[1]/(sinus) 
		n[2] = q[2]/(sinus)
	return n, theta 

def n_theta_to_Quat( n, theta, q=[0., 0., 0., 0]):
	sin_a = sin(theta / 2.0)
	cos_a = cos(theta / 2.0)

	q[0] = n[0] * sin_a
	q[1] = n[1] * sin_a
	q[2] = n[2] * sin_a
	q[3] = cos_a

	return normalize_Quat(q)

"""---------------------------------------------------------------
| Euler_to_Quat, Quat_to_Euler 	
| Description : Convertit un angle de Euler en quat et vis versa
 ---------------------------------------------------------------"""
def euler_to_Quat( e,  q=[0., 0., 0., 0] ):

	q[3] = ( cos(e[0]/2.0)*cos(e[1]/2.0)*cos(e[2]/2.0) + sin(e[0]/2.0)*sin(e[1]/2.0)*sin(e[2]/2.0) ) 
	q[0] = ( sin(e[0]/2.0)*cos(e[1]/2.0)*cos(e[2]/2.0) - cos(e[0]/2.0)*sin(e[1]/2.0)*sin(e[2]/2.0) ) 
	q[1] = ( cos(e[0]/2.0)*sin(e[1]/2.0)*cos(e[2]/2.0) + sin(e[0]/2.0)*cos(e[1]/2.0)*sin(e[2]/2.0) ) 
	q[2] = ( cos(e[0]/2.0)*cos(e[1]/2.0)*sin(e[2]/2.0) - sin(e[0]/2.0)*sin(e[1]/2.0)*cos(e[2]/2.0) ) 
	return normalize_Quat( q )

def quat_to_Euler(  q,  e=[0., 0., 0.] ):
	normalize_Quat(q)
	q0 = q[3], q1 = q[0], q2 = q[1], q3 = q[2]

	e[0] = atan2(2.0*(q0*q1 + q2*q3), (1.0 - 2.0*(q1*q1 + q2*q2)))
	e[1] = asin (2.0*(q0*q2 - q3*q1))
	e[2] = atan2(2.0*(q0*q3 + q2*q1), (1.0 - 2.0*(q2*q2 + q3*q3)))

	return e

"""---------------------------------------------------------------
| multiplication_Quat
| Description : multiplication d'hamilton
 ---------------------------------------------------------------"""
def multiplication_Quat( q1, q2,  res=[0., 0., 0., 0] ):

	res[0] = q1[3]*q2[0] + q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1]
	res[1] = q1[3]*q2[1] + q1[1]*q2[3] + q1[2]*q2[0] - q1[0]*q2[2]
	res[2] = q1[3]*q2[2] + q1[2]*q2[3] + q1[0]*q2[1] - q1[1]*q2[0]
	res[3] = q1[3]*q2[3] - q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2]

	return normalize_Quat( res )

"""---------------------------------------------------------------
| conjugue_Quat
| Description : conjugue
 ---------------------------------------------------------------"""
def conjugue_Quat( q,  res=[0., 0., 0., 0] ):

	res[0] = -q[0]
	res[1] = -q[1]
	res[2] = -q[2]
	res[3] =  q[3]

	return res 

"""---------------------------------------------------------------
| inverse_Quat
| Description : inverse 
 ---------------------------------------------------------------"""
def inverse_Quat( q,  res=[0., 0., 0., 0] ) :

	conjugue_q[4] = 0.0, 0.0, 0.0, 0.0
	n  = norm_Quat(q)
	conjugue_Quat(q, conjugue_q)

	res[0] = conjugue_q[0]/n
	res[1] = conjugue_q[1]/n
	res[2] = conjugue_q[2]/n
	res[3] = conjugue_q[3]/n

	return res

"""---------------------------------------------------------------
| fromMatrix
| Description : convertie une matrice de rotation en quaternion
 ---------------------------------------------------------------"""
def fromMatrix( m, q=[0., 0., 0., 0.]) :

	tr = m[0] + m[4] + m[8]

	# check the diagonal
	if (tr > 0):	
		s = sqrt (tr + 1)
		q[3] = s * 0.5 # w OK
		s = 0.5 / s
		q[0] = (m[7] - m[5]) * s # x OK
		q[1] = (m[2] - m[6]) * s # y OK
		q[2] = (m[3] - m[1]) * s # z OK
	
	else :	
		if (m[4] > m[0] and m[8] <= m[4]):		
			s = sqrt ((m[4] - (m[8] + m[0])) + 1.0)
			q[1] = s * 0.5 # y OK

			if (s != 0.0):
				s = 0.5 / s

			q[2] = (m[5] + m[7]) * s # z OK
			q[0] = (m[1] + m[3]) * s # x OK
			q[3] = (m[2] - m[6]) * s # w OK
		
		elif ((m[4] <= m[0]  and  m[8] > m[0])  or  (m[8] > m[4])):
		
			s = sqrt ((m[8] - (m[0] + m[4])) + 1.0)

			q[2] = s * 0.5 # z OK

			if (s != 0.0):
				s = 0.5 / s

			q[0] = (m[6] + m[2]) * s # x OK
			q[1] = (m[5] + m[7]) * s # y OK
			q[3] = (m[3] - m[1]) * s # w OK
		
		else:		
			s = sqrt ((m[0] - (m[4] + m[8])) + 1.0)
			q[0] = s * 0.5 # x OK

			if (s != 0.0):
				s = 0.5 / s

			q[1] = (m[1] + m[3]) * s # y OK
			q[2] = (m[6] + m[2]) * s # z OK
			q[3] = (m[7] - m[5]) * s # w OK
		
	
	normalize_Quat(q)
	return q

"""---------------------------------------------------------------
| toMatrix
| Description : convertie un quaternion en matrice de rotation
 ---------------------------------------------------------------"""
def toMatrix(q, m):

	wx, wy, wz, xx, yy, yz, xy, xz, zz

	xx = 2 * q[0] * q[0];   xy = 2 * q[0] * q[1];   xz = 2 * q[0] * q[2]
	yy = 2 * q[1] * q[1];   yz = 2 * q[1] * q[2];   zz = 2 * q[2] * q[2]
	wx = 2 * q[3] * q[0];   wy = 2 * q[3] * q[1];   wz = 2 * q[3] * q[2]

	m[0] = 1 - yy - zz;  m[3] = xy - wz    ;  m[6] = xz + wy
	m[1] = xy + wz    ;  m[4] = 1 - xx - zz;  m[7] = yz - wx
	m[2] = xz - wy    ;  m[5] = yz + wx    ;  m[8] = 1 - xx - yy

	return m


"""---------------------------------------------------------------
| compute_orientation_from_two_points - calcule un quaternion
| suivant l'orientation d'une ligne definie par 2 points.
| - xyz = represent the fact that v1 = p1p2/norm(p1p2) is the i^th
| vector of the frame
 ---------------------------------------------------------------"""
def computeQuatFromPoints(P1, P2, sign=1, xyz=1):
    v1 = VECT3_mult_by_coef(sign, VECT3_normalize(VECT3_diff(P1, P2)))
    v2 = VECT3_normalize([v1[1], -v1[0], 0])
    v3 = VECT3_normalize(VECT3_cross(v1, v2))
    if(xyz==1) :
        m = [v1[0], v2[0], v3[0], v1[1], v2[1], v3[1], v1[2], v2[2], v3[2]]
    if(xyz==2) :
        m = [v3[0], v1[0], v2[0], v3[1], v1[1], v2[1], v3[2], v1[2], v2[2]]
    if(xyz==3) :
        m = [v2[0], v3[0], v1[0], v2[1], v3[1], v1[1], v2[2], v3[2], v1[2]]
    q = fromMatrix(m)
    return q

"""---------------------------------------------------------------"""
"""********************************************************************************************
|
|                                           Vecteurs
|
 ********************************************************************************************"""
"""---------------------------------------------------------------
| VECT3_norme - calcul de la norme d'un vecteur
 --------------------------------------------------------------"""
def VECT3_norm(v):
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

"""---------------------------------------------------------------
| VECT3_norme - calcul de la norme d'un vecteur
 --------------------------------------------------------------"""
def VECT3_normalize(v):
    norm_v = VECT3_norm(v)
    return [v[0]/norm_v, v[1]/norm_v, v[2]/norm_v]

"""---------------------------------------------------------------
| VECT3_change_norm - Change la norme d'un vecteur
 --------------------------------------------------------------"""
def VECT3_change_norm( input,  r,  output=[0.0, 0.0, 0.0]):
	norm_input = VECT3_norm(input)

	if(norm_input > 0.0):
		for i in range(0, 3):
			output[i] = input[i]*(r/norm_input)
		return output
	else :
		for i in range(0, 3):
			output[i] = input[i]

	return output

"""---------------------------------------------------------------
| VECT3_dist - Calcule la distance entre les vecteurs v1 et v2
 --------------------------------------------------------------"""
def VECT3_dist( v1,  v2):
    dist = 0.0
    v1_2 = [(v1[0]-v2[0]), (v1[1]-v2[1]), (v1[2]-v2[2])]        
    return VECT3_norm(v1_2)

"""---------------------------------------------------------------
| VECT3_mult_par_scalaire - multiplication d'un vecteur par un scalaire
 --------------------------------------------------------------"""
def VECT3_mult_by_coef( coef,  v):

    x = coef * v[0]
    y = coef * v[1]
    z = coef * v[2]

    return [x, y, z]

"""---------------------------------------------------------------
| VECT3_saturateNorm - saturation de la norme du vecteur 
 ---------------------------------------------------------------"""
def VECT3_saturateNorm ( vec, maxNorm):

  vecNorm = VECT3_norm(vec)

  if ((vecNorm != 0.0) and (vecNorm > maxNorm)) :
    VECT3_change_norm(vec, maxNorm, vec)

"""---------------------------------------------------------------
| VECT3_mult_par_matrice - multiplication d'un vecteur par une 
|						   matrice 3*3
 ---------------------------------------------------------------"""
def VECT3_mult_par_matrice (mat, v,  res=[0.0, 0.0, 0.0]):
    for i in range(0,3):	
        res[i] = 0
        for j in range(0,3):
            res[i] += mat[3*i + j] * v[j]

    return (res)

"""---------------------------------------------------------------
| VECT3_produit_vectoriel - produit vectoriel de v1 et v2
 ---------------------------------------------------------------"""
def VECT3_cross(v1, v2):
    x = (v1[1] * v2[2]) - (v1[2] * v2[1])
    y = (v1[2] * v2[0]) - (v1[0] * v2[2])
    z = (v1[0] * v2[1]) - (v1[1] * v2[0])
    return [x, y, z]

"""---------------------------------------------------------------
| VECT3_equal - egalite de v1 et v2
 ---------------------------------------------------------------"""
def VECT3_equal(v1, v2, err=0.00001):	
    if(VECT3_dist(v1,v2)<=err) : return True
    else : return False

"""---------------------------------------------------------------
| VECT3_somme - somme vectoriel de v1 et v2
 ---------------------------------------------------------------"""
def VECT3_sum(v1, v2):
	
	x = v1[0] + v2[0] 
	y = v1[1] + v2[1] 
	z = v1[2] + v2[2] 

	return [x, y, z]

"""---------------------------------------------------------------
| VECT3_somme - somme vectoriel de v1 et v2
 ---------------------------------------------------------------"""
def VECT3_diff(v1, v2):
	
	x = v1[0] - v2[0] 
	y = v1[1] - v2[1] 
	z = v1[2] - v2[2] 

	return [x, y, z]

"""---------------------------------------------------------------
| VECT3_toMatrix - convertit un vecteur en matrice de rotation
 ---------------------------------------------------------------"""
def VECT3_toMatrix( u,  theta, res=[0., 0., 0., 0., 0., 0., 0., 0., 0.]):
    c = cos(theta)
    s = sin(theta)
    ux = u[0];	uy = u[1];	uz = u[2]

    res[0] = ux*ux + (1.0 - ux*ux)*c;	res[1] = ux*uy*(1.0 - c) - uz*s ;	res[2] = ux*uz*(1.0 - c) + uy*s  
    res[3] = ux*uy*(1.0 - c) + uz*s ;	res[4] = uy*uy + (1.0 - uy*uy)*c;	res[5] = uy*uz*(1.0 - c) - ux*s  
    res[6] = ux*uz*(1.0 - c) - uy*s ;	res[7] = ux*uz*(1.0 - c) + ux*s ;	res[8] = uz*uz + (1.0 - uz*uz)*c 

    return res

"""---------------------------------------------------------------
| VECT3_minDist - calcule la distance minimal entre le point p 
| et un mesh M
 ---------------------------------------------------------------"""
def VECT3_minDist(v, M):
    ind = 0
    min = VECT3_dist(v, M[0])
    for p in M : 
        d = VECT3_dist(p, v)
        if(d < min) :
            min = d
            ind = M.index(p)
    return (min, ind)

"""---------------------------------------------------------------
| VECT3_translation_VECT3 : translation d'un vecteur v par un vecteur t
 ---------------------------------------------------------------"""
def VECT3_translation_VECT3( v,  t,  res=[0., 0., 0.]):
	VECT3_somme(v, t, res)
	return res 

"""---------------------------------------------------------------
| VECT3_rotate_quat - rotation de v à l'aide du quaternion q
 ---------------------------------------------------------------"""
def VECT3_rotate_quat( v,  q):
	a=q[3]; b=q[0]; c=q[1]; d=q[2]
	t2 =   a*b
	t3 =   a*c
	t4 =   a*d
	t5 =  -b*b
	t6 =   b*c
	t7 =   b*d
	t8 =  -c*c
	t9 =   c*d
	t10 = -d*d
	x = 2*( (t8 + t10)*v[0] + (t6 -  t4)*v[1] + (t3 + t7)*v[2] ) + v[0]
	y = 2*( (t4 +  t6)*v[0] + (t5 + t10)*v[1] + (t9 - t2)*v[2] ) + v[1]
	z = 2*( (t7 -  t3)*v[0] + (t2 +  t9)*v[1] + (t5 + t8)*v[2] ) + v[2]

	return [x, y, z] 

"""---------------------------------------------------------------
| VECT3_rotate_u_theta - rotation de theta de v autour de l'axe u
 ---------------------------------------------------------------"""
def VECT3_rotate_u_theta(v, u, theta):
	ux = u[0]
	uy = u[1]
	uz = u[2]
	c = cos(theta)
	s = sin(theta)

	r00 = ux*ux + (1.0 - ux*ux)*c;	r01 = ux*uy*(1.0 - c) - uz*s ;	r02 = ux*uz*(1.0 - c) + uy*s  	
	r10 = ux*uy*(1.0 - c) + uz*s ;	r11 = uy*uy + (1.0 - uy*uy)*c;	r12 = uy*uz*(1.0 - c) - ux*s  
	r20 = ux*uz*(1.0 - c) - uy*s ;	r21 = ux*uz*(1.0 - c) + ux*s ;	r22 = uz*uz + (1.0 - uz*uz)*c 

	x = r00*v[0] + r01*v[1] + r02*v[2] 
	y = r10*v[0] + r11*v[1] + r12*v[2] 
	z = r20*v[0] + r21*v[1] + r22*v[2] 

	return [x, y, z] 

"""---------------------------------------------------------------
| VECT3_rotate_u_theta_A - rotation de theta de G autour de l'axe u 
|						   passant par A
 ---------------------------------------------------------------"""
def VECT3_rotate_u_theta_A(G, A, u,  theta):
	AG	 = [0.0, 0.0, 0.0]
	
	for i in range(0, 3): 
		AG[i] = G[i] - A[i]

	r_AG = VECT3_rotate_u_theta(AG, u, theta)

	x = A[0] + r_AG[0]
	y = A[1] + r_AG[1]
	z = A[2] + r_AG[2]

	return [x, y, z]

"""---------------------------------------------------------------
| VECT3_produit_scalaire - produit scalaire de deux vecteurs
 ---------------------------------------------------------------"""
def VECT3_dot( v1,  v2):
	res = 0.0
	res = v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] 
	return res

"""---------------------------------------------------------------
| VECT3_middle - retourne le milieu d'un vecteur
 ---------------------------------------------------------------"""
def VECT3_middle(v1, v2):
	x = (v1[0]+v2[0])/2.0
	y = (v1[1]+v2[1])/2.0
	z = (v1[2]+v2[2])/2.0 
	return [x, y, z]

"""---------------------------------------------------------------
| VECT3_orthogonal_projection - retourne le projete ortho de A sur
| sur [P1P2]
 ---------------------------------------------------------------"""
def VECT3_orthogonal_projection( A,  P1, P2, accurency=0.01):
	P1P2 = VECT3_normalize(VECT3_diff(P2, P1))
	A_prime = P1[:]
	cpt = 0
	while VECT3_produit_scalaire(VECT3_diff(A, A_prime), P1P2)>1e-8 :
	    A_prime = VECT3_somme(P1, VECT3_mult_par_scalaire((cpt*accurency), P1P2))
	    cpt = cpt + 1
        return A_prime
"""---------------------------------------------------------------
| clockWise - retourne if polygon CW or CCW
 ---------------------------------------------------------------"""
def clockWise(p) :
   count = 0
   n = len(p)
   if (n < 3): return 0

   for i in range(n) :
      j = (i + 1) % n
      k = (i + 2) % n
      z = (p[j][0] - p[i][0]) * (p[k][1] - p[j][1])
      z = z - (p[j][1] - p[i][1]) * (p[k][0] - p[j][0])
      if z<0:
         count = count - 1
      elif z>0:
         count = count + 1
   
   if count > 0:    return 1
   elif count < 0:  return -1
   else: return 0

"""---------------------------------------------------------------
| POLYGON_equal - true if the two polygon avec the same indice
 ---------------------------------------------------------------"""
def POLYGON_equal(POLY1, POLY2) :
    # Check if it's possible to find an egality
    if(len(POLY1)!=len(POLY2)) :
        return false;
    # Init the bool
    bool = []
    for i in range(0, len(POLY1)): 
        bool.append(False)
    # Check if the polygon are equal
    for i in range(0, len(POLY1)):
        i1 = POLY1[i]
        for i2 in POLY2:
            if(i1==i2): 
                bool[i]=True; 
                break
    for b in bool: 
        if not (b): 
            return False
    return True    

"""---------------------------------------------------------------
| Convert a tab into a list
 ---------------------------------------------------------------"""
def toStr(tab, level=2) :
    output = ''
    if (level == 1) :
        for e_i in tab :
            output = output + ' ' + str(e_i)
    elif (level == 2):
        for e in tab :
            for e_i in e :
                output = output + ' ' + str(e_i)
    return output



"""---------------------------------------------------------------
| Compute the normal of a triangle
 ---------------------------------------------------------------"""
def computeTriangleNormal(p1, p2, p3) : 
    n = VECT3_cross(VECT3_diff(p1,p2), VECT3_diff(p1,p3))
    n = VECT3_normalize(n)
    return n

def computeQuadNormal(p1, p2, p3, p4) : 
    n = VECT3_cross(VECT3_diff(p1,p2), VECT3_diff(p1,p4))
    n = VECT3_normalize(n)
    return n

def computeNormal(pts, poly, invertNormal=-1) :
    N = [ [0,0,0] ]*len(pts)
    # Computation of normals
    for f in poly : 
        n = []
        if (len(f)==3):
            n = computeTriangleNormal(pts[f[0]], pts[f[1]], pts[f[2]])
        if (len(f)==4):
            n = computeQuadNormal(pts[f[0]], pts[f[1]], pts[f[2]], pts[f[3]])
        for i in f :
            N[i] = VECT3_sum(N[i],n)

    # Normalisation
    for n in N : 
        i = N.index(n)
        N[i] = VECT3_normalize(N[i])

    # Reorientation of normal
    for i in range(0, (len(N)-1)) : 
        n1 = N[i]
        n2 = N[i+1]
        if ( VECT3_dot(n1,n2) < 0 ) :
            N[i+1] = VECT3_mult_by_coef(-1.0, N[i+1])

    # Inversion of normal if it is necessary
    if (invertNormal==1) : 
        for n in N : 
            i = N.index(n)
            N[i] = VECT3_mult_by_coef(-1, N[i])
    return N

"""--------------------------------------------------------------"""

