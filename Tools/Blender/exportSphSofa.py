# export des spheres dans un fichier sph pour SOFA

import struct
from struct import *
import array
from array import *
import Blender

import math
from math import *

from Blender import NMesh
from Blender.Draw import *

#poly = NMesh.GetRaw('Sphere')
#polyObj = NMesh.PutRaw(poly)
###polyObj.setLocation (0.0, 0.0, 0.0)
#Blender.Redraw()

objects = Blender.Object.GetSelected()
print objects
	
def writeLong (fsph, integ):
	c = unpack('4b', pack ('l', integ))
	
	for i in range (4):
		#print "d = "+`c[i]`
		fsph.write(pack('b', c[i]))

def writeFloat (fsph, x):
	c = unpack('4b', pack ('f', x))
	#c = unpack ('4b', x)
	
	for i in range (4):
		#print "d = "+`c[i]`
		fsph.write(pack  ('b', c[i]))

def writeSph (fsph, obj,i):
	location = obj.getLocation()
	#print ("une sphere en : "+`location`)
	#r = obj.SizeX #le scale est forcement uniforme pour la sphere sinon cela ne fonctionne pas pour SPORE
	#me = NMesh.GetRawFromObject(obj.name)
	
	#v = me.verts[0]
	#print `v.co[0]`
	#r = sqrt ((v.co[0] - location[0]) ** 2 + (v.co[1] - location[1])** 2 + (v.co[2] - location[2])** 2)
	#print "r = "+ `r`
	r = obj.SizeX
	fsph.write('sphe %s %s %s %s %s\n' % (i, location[0], location[1], location[2],r))	
	
def exportSPH(fsph):
	print("writing")
	fsph.write("sph 1.0\n")
	nbSph = len(objects)
	print "Nombre de sph : "+`nbSph`	
	fsph.write('nums %s\n' % (nbSph))
	i=1
	for obj in objects:
		writeSph(fsph, obj,i)
		i=i+1
	
fsph = file ("D:/coussin.sph", "w")
#fsph = file ("/homelocal/fonteneau/tmpSimdev/simdev/simdyn/SUPOTEST/OBJETS/ovaire_droit2.sph", "wb")
exportSPH(fsph)
