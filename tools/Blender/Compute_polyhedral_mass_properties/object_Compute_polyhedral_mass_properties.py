#!BPY
"""
Name: 'Compute polyhedral mass properties'
Blender: 243
Group: 'Object'
Tooltip: 'Compute position of center of mass and other mass properties'
"""

__author__ = "Brian Mirtich (adaptation à Blender par Vincent Vansuyt en 2008) "
__url__ = ("http://www.cs.berkeley.edu/~jfc/mirtich",  "http://www.vincentvansuyt.com/bld_Index.htm")
__version__ = "1.0 1995"

__bpydoc__ = """\
Ce script calcule la position du centre de gravité et des éléments d'inertie du maillage de l'objet sélectionné.
"""

#*******************************************************
#*                                                      *
#*  volInt.c                                            *
#*                                                      *
#*  This code computes volume integrals needed for      *
#*  determining mass properties of polyhedral bodies.   *
#*                                                      *
#*  For more information, see the accompanying README   *
#*  file, and the paper                                 *
#*                                                      *
#*  Brian Mirtich, "Fast and Accurate Computation of    *
#*  Polyhedral Mass Properties," journal of graphics    *
#*  tools, volume 1, number 1, 1996.                    *
#*                                                      *
#*  This source code is public domain, and may be used  *
#*  in any way, shape or form, free of charge.          *
#*                                                      *
#*  Copyright 1995 by Brian Mirtich                     *
#*                                                      *
#*  mirtich@cs.berkeley.edu                             *
#*  http://www.cs.berkeley.edu/~mirtich                 *
#*                                                      *
#*******************************************************/
#
# [VVansuyt] Extraits du README de Brian Mirtich (j'ai mis à jour les paragraphes après
#  les notations "[VVansuyt]" )
#
#************************************************************************************************
# 1.  OVERVIEW
#	This code accompanies the paper:
#
#	Brian Mirtich, "Fast and Accurate Computation of
#	Polyhedral Mass Properties," journal of graphics
#	tools, volume 1, number 2, 1996.
#
#	It computes the ten volume integrals needed for
#	determining the center of mass, moments of
#	inertia, and products of inertia for a uniform
#	density polyhedron.  From this information, a
#	body frame can be computed.
#
#	To compile the program, use an ANSI compiler, and
#	type something like
# 
#		% cc volInt.c -O2 -lm -o volInt
#
# [VVansuyt] (plus la peine de compiler avec Blender et Python)  
#
#
#	Revision history
#
#	26 Jan 1996	Program creation.
#
#	 3 Aug 1996	Corrected bug arising when polyhedron density
#			is not 1.0.  Changes confined to function main().
#			Thanks to Zoran Popovic for catching this one.
#
#	[VVansuyt] 17 June 2008	portage du code "VolInt.C" pour Python et Blender
#
#
# 2.  POLYHEDRON GEOMETRY FILES
#	[VVansuyt] Plus besoin d'utiliser des fichiers au format "Polyhedron", c'est le format de 
#		maillage de Blender qui est utilisé en entrée du programme
#
#
# 3.  RUNNING THE PROGRAM
#	[VVansuyt] Into Blender, choose an object (right clic)
#	Press the TAB key to enter in edition mode
#	Call this program by using the Blender "Mesh" menu 
#
#
# 4.  DISCLAIMERS
#
#	1.  The volume integration code has been written
#	to match the development and algorithms presented
#	in the paper, and not with maximum optimization
#	in mind.  While inherently very efficient, a few
#	more cycles can be squeezed out of the algorithm.
#	This is left as an exercise. :)
#
#	2.  Don't like global variables?  The three
#	procedures which evaluate the volume integrals
#	can be combined into a single procedure with two
#	nested loops.  In addition to providing some
#	speedup, all of the global variables can then be
#	made local.
#
#	3.  The polyhedron data structure used by the
#	program is admittedly lame; much better schemes
#	are possible.  The idea here is just to give the
#	basic integral evaluation code, which will have
#	to be adjusted for other polyhedron data
#	structures.
#
#	4.  There is no error checking for the input
#	files.  Be careful.  Note the hard limits
#	#defined for the number of vertices, number of
#	faces, and number of vertices per faces.
#

import Blender
from Blender import Draw, BGL, Window, Scene, NMesh, sys, Object
import math
from math import sqrt, floor
import BPyMesh
import BPyMessages
import platform
import os 

# ============================================================================
# constants
# ============================================================================

X = 0
Y = 1
Z = 2

# ============================================================================
# data structures
# ============================================================================

class FACE():
	
	def __init__( self ):
		self.numVerts = 0
		self.norm = [ 0.0, 0.0, 0.0 ]
		self.w = 0.0
		self.verts = [] # tableau d'entier des indices de sommet
		
	def afficher( self ):
		print self.verts
		print self.norm
		print self.w


class POLYHEDRON():
	
	def __init__( self ):
		self.numVerts = 0
		self.numFaces = 0
		self.verts = [[]] # 3 doubles
		self.faces = []
		
	def afficher( self ):
		print "POLYHEDRON::afficher()"
		print "%d sommet(s)"%(self.numVerts)
		i = 0
		while i < self.numVerts:
			print " (%.3f, %.3f, %.3f)"%(self.verts[ i ][0], self.verts[ i ][1], self.verts[ i ][2])
			i += 1
		
		print "%d face(s)"%(self.numFaces)
		i = 0
		for face in self.faces:
			print "Face %d, nb de sommets : %d"%( i, face.numVerts )
			face.afficher()
			i += 1
		

#============================================================================
# globals
#============================================================================

A = B = C = 0 # alpha, beta, gamma

# projection integrals 
P1 = Pa = Pb = Paa = Pab = Pbb = Paaa = Paab = Pabb = Pbbb = 0.0

#/* face integrals */
Fa = Fb = Fc = Faa = Fbb = Fcc = Faaa = Fbbb = Fccc = Faab = Fbbc = Fcca = 0.0

#/* volume integrals */
T0 = 0.0
T1 = [0.0, 0.0, 0.0]
T2 = [0.0, 0.0, 0.0]
TP = [0.0, 0.0, 0.0]


#============================================================================
# read in a polyhedron
# Lit les données du maillage reçu, peuple un objet POLYHEDRON à partir du
#  maillage reçu et le renvoie
#============================================================================
def readPolyhedron( maillage ): # char * name, POLYHEDRON p
	
	# Instanciation du polyhedron de retour
	p = POLYHEDRON()
	
	# Parcours des données du maillage reçu et remplissage du polyhedron
	# - récupération de la liste des coordonnées des sommets
	p.numVerts = len( maillage.verts )
	p.verts = []
	for v in maillage.verts:
		#print v
		p.verts.append( [ v.co[0], v.co[1], v.co[2] ] )
	
	# - récupération des informations sur les faces
	nbFaces = len( maillage.faces )
	p.numFaces = nbFaces
	p.faces = []
	for face in maillage.faces:
		#print face
		# Récupération des informations pour la face
		f = FACE()
		nbSommets = len(face.verts)
		f.numVerts = nbSommets
		for vertex in face.verts:
			f.verts.append( vertex.index )
		# Remplissage des informations de la face
		#/* compute face normal and offset w from first 3 vertices */
		dx1 = p.verts[f.verts[1]][X] - p.verts[f.verts[0]][X];
		dy1 = p.verts[f.verts[1]][Y] - p.verts[f.verts[0]][Y];
		dz1 = p.verts[f.verts[1]][Z] - p.verts[f.verts[0]][Z];
		dx2 = p.verts[f.verts[2]][X] - p.verts[f.verts[1]][X];
		dy2 = p.verts[f.verts[2]][Y] - p.verts[f.verts[1]][Y];
		dz2 = p.verts[f.verts[2]][Z] - p.verts[f.verts[1]][Z];
		nx = dy1 * dz2 - dy2 * dz1;
		ny = dz1 * dx2 - dz2 * dx1;
		nz = dx1 * dy2 - dx2 * dy1;
		len_b = sqrt(nx * nx + ny * ny + nz * nz);
		f.norm[X] = nx / len_b;
		f.norm[Y] = ny / len_b;
		f.norm[Z] = nz / len_b;
		f.w = - f.norm[X] * p.verts[f.verts[0]][X]\
				 - f.norm[Y] * p.verts[f.verts[0]][Y]\
				 - f.norm[Z] * p.verts[f.verts[0]][Z]
		# Ajout de la face dans la liste des faces du polyhedron et la liste des sommets
		p.faces.append( f )
		
	#p.afficher()
	return p



# ============================================================================
# compute mass properties
# ============================================================================

#/* compute various integrations over projection of face */
def compProjectionIntegrals( f, p ):# FACE *f
	global P1, Pa, Pb, Paa, Pab, Pbb, Paaa, Paab, Pabb, Pbbb
	P1 = Pa = Pb = Paa = Pab = Pbb = Paaa = Paab = Pabb = Pbbb = 0.0;
	global A, B, C

	i = 0
	while i < f.numVerts:
		a0 = p.verts[f.verts[i]][A]
		b0 = p.verts[f.verts[i]][B]
		a1 = p.verts[f.verts[(i+1) % f.numVerts]][A]
		b1 = p.verts[f.verts[(i+1) % f.numVerts]][B]
		da = a1 - a0;
		db = b1 - b0;
		a0_2 = a0 * a0; a0_3 = a0_2 * a0; a0_4 = a0_3 * a0;
		b0_2 = b0 * b0; b0_3 = b0_2 * b0; b0_4 = b0_3 * b0;
		a1_2 = a1 * a1; a1_3 = a1_2 * a1; 
		b1_2 = b1 * b1; b1_3 = b1_2 * b1;

		C1 = a1 + a0;
		Ca = a1*C1 + a0_2; Caa = a1*Ca + a0_3; Caaa = a1*Caa + a0_4;
		Cb = b1*(b1 + b0) + b0_2; Cbb = b1*Cb + b0_3; Cbbb = b1*Cbb + b0_4;
		Cab = 3*a1_2 + 2*a1*a0 + a0_2; Kab = a1_2 + 2*a1*a0 + 3*a0_2;
		Caab = a0*Cab + 4*a1_3; Kaab = a1*Kab + 4*a0_3;
		Cabb = 4*b1_3 + 3*b1_2*b0 + 2*b1*b0_2 + b0_3;
		Kabb = b1_3 + 2*b1_2*b0 + 3*b1*b0_2 + 4*b0_3;
		
		P1 += db*C1
		Pa += db*Ca
		Paa += db*Caa
		Paaa += db*Caaa
		Pb += da*Cb
		Pbb += da*Cbb
		Pbbb += da*Cbbb
		Pab += db*(b1*Cab + b0*Kab)
		Paab += db*(b1*Caab + b0*Kaab)
		Pabb += da*(a1*Cabb + a0*Kabb)
		# suiv
		i += 1

	P1 /= 2.0
	Pa /= 6.0
	Paa /= 12.0
	Paaa /= 20.0
	Pb /= -6.0
	Pbb /= -12.0
	Pbbb /= -20.0
	Pab /= 24.0
	Paab /= 60.0
	Pabb /= -60.0


def compFaceIntegrals( f, p ):
	global P1, Pa, Pb, Paa, Pab, Pbb, Paaa, Paab, Pabb, Pbbb
	global Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca
	global A, B, C
	
	compProjectionIntegrals( f, p )
	
	w = f.w;
	n = f.norm;
	k1 = 1.0 / n[C];
	k2 = k1 * k1;
	k3 = k2 * k1;
	k4 = k3 * k1;
	
	Fa = k1 * Pa;
	Fb = k1 * Pb;
	Fc = -k2 * (n[A]*Pa + n[B]*Pb + w*P1);

	Faa = k1 * Paa;
	Fbb = k1 * Pbb;
	Fcc = k3 * (pow(n[A],2)*Paa + 2*n[A]*n[B]*Pab + pow(n[B],2)*Pbb
	 + w*(2*(n[A]*Pa + n[B]*Pb) + w*P1));

	Faaa = k1 * Paaa;
	Fbbb = k1 * Pbbb;
	Fccc = -k4 * (pow(n[A],3)*Paaa + 3*pow(n[A],2)*n[B]*Paab \
				+ 3*n[A]*pow(n[B],2)*Pabb + pow(n[B],3)*Pbbb \
				+ 3*w*(pow(n[A],2)*Paa + 2*n[A]*n[B]*Pab + pow(n[B],2)*Pbb) \
				+ w*w*(3*(n[A]*Pa + n[B]*Pb) + w*P1))
	
	Faab = k1 * Paab;
	Fbbc = -k2 * (n[A]*Pabb + n[B]*Pbbb + w*Pbb);
	Fcca = k3 * (pow(n[A], 2)*Paaa + 2*n[A]*n[B]*Paab + pow(n[B], 2)*Pabb\
		+ w*(2*(n[A]*Paa + n[B]*Pab) + w*Pa));


def compVolumeIntegrals( p ):# POLYHEDRON *p
	
	global T0, T1, T2, TP
	global A, B, C
	global Fa, Fb, Fc, Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca
	
	T0 = 0
	T1 = [ 0.0, 0.0 , 0.0 ]
	T2 = [ 0.0, 0.0 , 0.0 ]
	TP = [ 0.0, 0.0 , 0.0 ]
	
	i = 0
	while ( i < p.numFaces ):
		f = p.faces[i];
		
		nx = abs(f.norm[X]);
		ny = abs(f.norm[Y]);
		nz = abs(f.norm[Z]);
		if (nx > ny and nx > nz):
			C = X;
		else:
			if ny > nz:
				C = Y
			else:
				C = Z
		A = (C + 1) % 3;
		B = (A + 1) % 3;
		
		compFaceIntegrals( f, p );
		
		mul = 1
		if A == X :
			mul = Fa
		else:
			if B == X:
				mul = Fb
			else:
				mul = Fc
		#T0 += f.norm[X] * ((A == X) ? Fa : ((B == X) ? Fb : Fc));
		T0 += f.norm[X] * mul
		
		global Faa, Fbb, Fcc, Faaa, Fbbb, Fccc, Faab, Fbbc, Fcca
		T1[A] += f.norm[A] * Faa;
		T1[B] += f.norm[B] * Fbb;
		T1[C] += f.norm[C] * Fcc;
		
		T2[A] += f.norm[A] * Faaa;
		T2[B] += f.norm[B] * Fbbb;
		T2[C] += f.norm[C] * Fccc;
		
		TP[A] += f.norm[A] * Faab;
		TP[B] += f.norm[B] * Fbbc;
		TP[C] += f.norm[C] * Fcca;
		
		# suiv
		i += 1
	
	
	T1[X] /= 2;
	T1[Y] /= 2;
	T1[Z] /= 2;
	
	T2[X] /= 3;
	T2[Y] /= 3;
	T2[Z] /= 3;
	
	TP[X] /= 2;
	TP[Y] /= 2;
	TP[Z] /= 2;
	


def computeMassProperties( maillage, masse_volumique ):
	mass = 0.0
	r = [ 0.0, 0.0, 0.0 ] #  center of mass 
	J = [ [0.0,0.0,0.0], [0.0,0.0,0.0], [0.0,0.0,0.0] ] # /* matrice d'inertie */
	
	p = readPolyhedron( maillage );
	
	compVolumeIntegrals( p );
	
	global T0, T1, T2, TP
	
	mass = masse_volumique * T0;
	
	#/* compute center of mass */
	r[X] = T1[X] / T0;
	r[Y] = T1[Y] / T0;
	r[Z] = T1[Z] / T0;
	
	#/* compute inertia tensor */
	J[X][X] = masse_volumique * (T2[Y] + T2[Z]);
	J[Y][Y] = masse_volumique * (T2[Z] + T2[X]);
	J[Z][Z] = masse_volumique * (T2[X] + T2[Y]);
	J[X][Y] = J[Y][X] = - masse_volumique * TP[X];
	J[Y][Z] = J[Z][Y] = - masse_volumique * TP[Y];
	J[Z][X] = J[X][Z] = - masse_volumique * TP[Z];

	#/* translate inertia tensor to center of mass */
	J[X][X] -= mass * (r[Y]*r[Y] + r[Z]*r[Z]);
	J[Y][Y] -= mass * (r[Z]*r[Z] + r[X]*r[X]);
	J[Z][Z] -= mass * (r[X]*r[X] + r[Y]*r[Y]);
	J[Y][X] += mass * r[X] * r[Y] 
	J[X][Y] = J[Y][X]
	J[Z][Y] += mass * r[Y] * r[Z] 
	J[Y][Z] = J[Z][Y]
	J[X][Z] += mass * r[Z] * r[X] 
	J[Z][X] = J[X][Z]
	
	return mass, r, T0, T1, T2, TP, J
	
	
	
def appel_ui_masse_volumique():
	Draw.Register( gui_masse_volumique, None, button_event_masse_volumique)
	
	
def gui_masse_volumique():
	# Création d'Id pour les évènements
	nCpt_evt = 1
	global EV_BT_OK
	EV_BT_OK = nCpt_evt; nCpt_evt += 1
	global EV_BT_ANNULER
	EV_BT_ANNULER = nCpt_evt; nCpt_evt += 1
	global EV_MASS_VOL
	EV_MASS_VOL = nCpt_evt; nCpt_evt += 1
	
	pos_x = 5
	pos_y = 100; pas_y = 36
	BGL.glRasterPos2i( pos_x, pos_y )
	Draw.Text("Choisissez la masse volumique du maillage (et validez par \"OK\") :")
	
	#[nom bouton] = Draw.Number("[nom]", [numéro d'événement], [position x], [position y], \
	#                            [largeur], [hauteur], \
	#                            [valeur initiale], [valeur minimale],[valeur maximale], "[astuce]")
	global bt_masse_volumique
	largeur_mv = 360
	hauteur = 25
	pos_y -= pas_y
	bt_masse_volumique = Draw.Number("Masse volumique :", EV_MASS_VOL, pos_x, pos_y, \
								largeur_mv, hauteur, \
							1.0, 0.0, 100000.0, "saisissez la masse volumique")
	largeur_bt = 80
	pos_y -= pas_y
	dx_espacement = int( ( largeur_mv - 2.0 * largeur_bt ) / 3.0 )
	dx_espacement = int( dx_espacement + 0.5 )
	pos_x_bt_OK = pos_x + dx_espacement
	pos_x_bt_Annuler = pos_x_bt_OK + largeur_bt + dx_espacement
	Draw.PushButton("OK", EV_BT_OK, pos_x_bt_OK, pos_y, largeur_bt, hauteur, "Valide")
	Draw.PushButton("Annuler", EV_BT_ANNULER, pos_x_bt_Annuler, pos_y, largeur_bt, hauteur, "Annule")
	
	
	
def button_event_masse_volumique(evt):
	if evt == EV_BT_OK:
		#print "OK"
		#print "masse volumique = %0.3f"%( bt_masse_volumique.val )
		#print type(bt_masse_volumique )
		calculer_elements_maillage( bt_masse_volumique.val )
		Draw.Exit()
	
	if evt == EV_BT_ANNULER:
		print "Annuler"
		Draw.Exit()
	


def calculer_elements_maillage( masse_volumique ):
	# Récupération de l'objet actif et de son maillage
	scn = Scene.GetCurrent()
	act_obj = scn.objects.active
	if not act_obj or act_obj.type != 'Mesh':
		strM = "Vous devez sélectionner un objet de type \"maillage\""
		return Draw.PupMenu( strM )
	maillage = act_obj.getData( mesh = 1 )
	
	Window.WaitCursor( 1 )
	[masse, r, T0, T1, T2, TP, J] = computeMassProperties( maillage, masse_volumique )
	Window.WaitCursor( 0 )
	
	strFichierLog = "mass_properties.log"
	file = open( strFichierLog, "w")
	print "------ debut du calcul ------"
	print "\nElements d'inertie et centre de gravite pour le maillage \"%s\"\n\n"%( act_obj.getName() )
	file.write("\nElements d'inertie et centre de gravite pour le maillage \"%s\"\n\n"%( act_obj.getName() ))
		
	print "Masse =%+20.6f" %( masse )
	file.write( "Masse =%+20.6f\n" %( masse ) ) 
	
	print "Volume =%+19.6f" %( T0 );
	file.write( "Volume =%+19.6f\n" %( T0 ) )
	
	print "Masse volumique : %10.3f\n"%( masse_volumique )
	file.write( "Masse volumique =%+10.6f\n\n" %( masse_volumique ) )
	
	print "Tx  =  %+20.6f"   %( T1[X] )
	print "Ty  =  %+20.6f"   %( T1[Y] )
	print "Tz  =  %+20.6f\n" %( T1[Z] )
	
	print "Txx =  %+20.6f"   %( T2[X] )
	print "Tyy =  %+20.6f"   %( T2[Y] )
	print "Tzz =  %+20.6f\n" %( T2[Z] )
	
	print "Txy =  %+20.6f"   %( TP[X] )
	print "Tyz =  %+20.6f"   %( TP[Y] )
	print "Tzx =  %+20.6f\n" %( TP[Z] )
	
	file.write("Tx =   %+20.6f\n"   %( T1[X] ))
	file.write("Ty =   %+20.6f\n"   %( T1[Y] ))
	file.write("Tz =   %+20.6f\n\n" %( T1[Z] ))
	
	file.write("Txx =  %+20.6f\n"   %( T2[X] ))
	file.write("Tyy =  %+20.6f\n"   %( T2[Y] ))
	file.write("Tzz =  %+20.6f\n\n" %( T2[Z] ))
	
	file.write("Txy =  %+20.6f\n"   %( TP[X] ))
	file.write("Tyz =  %+20.6f\n"   %( TP[Y] ))
	file.write("Tzx =  %+20.6f\n\n" %( TP[Z] ))
	
	
	print "Centre de gravite :  (%+12.6f,%+12.6f,%+12.6f)\n" %(r[X], r[Y], r[Z])
	file.write( "Centre de gravite :  (%+12.6f,%+12.6f,%+12.6f)\n\n" %(r[X], r[Y], r[Z]) )
	
	pos_CDG_obj = act_obj.getLocation();
	Window.SetCursorPos( pos_CDG_obj[X] + r[X], pos_CDG_obj[Y] + r[Y], pos_CDG_obj[Z] + r[Z] )
	Blender.Redraw()
	
	print "Matrice d'inertie avec comme origine, le centre de gravite :\n";
	print " A = %+15.6f  -F = %+15.6f  -E = %+15.6f" %(J[X][X], J[X][Y], J[X][Z]);
	print "-F = %+15.6f   B = %+15.6f  -D = %+15.6f" %(J[Y][X], J[Y][Y], J[Y][Z]);
	print "-E = %+15.6f  -D = %+15.6f   C = %+15.6f\n" %(J[Z][X], J[Z][Y], J[Z][Z]);
	print " A = \int_{V}{ \left( y^2 + z^2 \\right) dm}"
	print " B = \int_{V}{ \left( x^2 + z^2 \\right) dm}"
	print " C = \int_{V}{ \left( x^2 + y^2 \\right) dm}"
	print " D = \int_{V}{ \left( y . z \\right) dm}"
	print " E = \int_{V}{ \left( x . z \\right) dm}"
	print " F = \int_{V}{ \left( x . y \\right) dm}"
	file.write( "Matrice d'inertie avec comme origine, le centre de gravité :\n" )
	file.write( "$I = \\left( \\begin{array}{rrrrrr} \n")
	file.write( " A =& %+15.6f & -F = & %+15.6f & -E =& %+15.6f \\\\ \n" %(J[X][X], J[X][Y], J[X][Z]) )
	file.write( "-F =& %+15.6f &  B = & %+15.6f & -D =& %+15.6f \\\\ \n" %(J[Y][X], J[Y][Y], J[Y][Z]) )
	file.write( "-E =& %+15.6f & -D = & %+15.6f &  C =& %+15.6f\n" %(J[Z][X], J[Z][Y], J[Z][Z]) )
	file.write( "\end{array} \\right)$ \n\n")
	file.write( "$$A = \int_{V}{ \left( y^2 + z^2 \\right) dm}$$\n" )
	file.write( "$$B = \int_{V}{ \left( x^2 + z^2 \\right) dm}$$\n" )
	file.write( "$$C = \int_{V}{ \left( x^2 + y^2 \\right) dm}$$\n" )
	file.write( "$$D = \int_{V}{ \left( y . z \\right) dm}$$\n" )
	file.write( "$$E = \int_{V}{ \left( x . z \\right) dm}$$\n" )
	file.write( "$$F = \int_{V}{ \left( x . y \\right) dm}$$\n" )
	
	print "\nLe curseur de Blender a ete deplace au centre de gravite du maillage."
	file.close()
	
	print "\nLa trace du resultat a ete enregistree dans le fichier \"%s\"."%(strFichierLog)
	
	if ( platform.system().lower() == "windows" ):
		os.startfile( strFichierLog )
	else:
		Draw.PupMenu("Fin du script|Consultez la console pour voir les resultats")
	
	print "------ fin du calcul ------"	
		

		
if __name__ == '__main__':
	# Vérification que l'on a bien sélectionné un objet de type "Maillage"
	scn = Scene.GetCurrent()
	act_obj = scn.objects.active
	if not act_obj or act_obj.type != 'Mesh':
		strM = "Attention|Vous devez sélectionner un objet de type \"maillage\""
		Draw.PupMenu( strM )
	else:
		# Effacement du texte de la console
		#print platform.system()
		if ( platform.system() == "Windows" ):#Linux
			os.system('cls') # pour Windows
		else:
			os.system('clear') # pour Linux
		appel_ui_masse_volumique( )
	
