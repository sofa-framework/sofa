import struct
from struct import *
import array
from array import *
import Blender

from Blender import Material
from Blender import NMesh
from Blender.Draw import *

# poly = NMesh.GetRaw('Sphere')
# polyObj = NMesh.PutRaw(poly)
### polyObj.setLocation (0.0, 0.0, 0.0)
# Blender.Redraw()

def readLong (fsph):
	c = array ('l', fsph.read(4))
	return c[0]

def readFloat (fsph):
	c = array('f', fsph.read(4)) 
	return c[0]

def drawSph (x, y, z, r):
	# mat = Material.New('myMat')
	# mat.rgbCol = [0.8, 0.1, 0.2]
	poly = NMesh.GetRaw('Sphere')
	polyObj = NMesh.PutRaw(poly)
	polyObj.setLocation (x, y, z)
	polyObj.SizeX = polyObj.SizeX * r
	polyObj.SizeY = polyObj.SizeY * r
	polyObj.SizeZ = polyObj.SizeZ * r
	print (polyObj.SizeX)
	# polyObj.addMaterial (mat)
	# print ("drawing sphere at "+`x`+", "+`y`+", "+`z`+", r = "+`r`)

	
def importSPH(fsph):
	print("opening")
 	fsph.readline()
 	Line = (fsph.readline()).split()
	if Line[0] == 'nums':				
		numSpheres = int(Line[1])
		print "Nombre de sph : "+`numSpheres`			
		for i in range (0, numSpheres):
			Line = (fsph.readline()).split()
			coorX = float(Line[2])
			coorY = float(Line[3])
			coorZ = float(Line[4])
			r = float(Line[5])
			print("coorX = " + Line[2] + ", coorY = " + Line[3] + ", coorZ = " + Line[4] + ",r = " + Line[5])
			drawSph (coorX, coorY, coorZ, r)
		 #Blender.Redraw()
	else:
		print("format de fichier incorrect")
	
# fsph = open("/homelocal/fonteneau/tmpSimdev/simdev/simdyn/MISS/OBJETS/SPICGYN2/uterus2.sph", "rb")
fsph= open("D:/Sofa_Save_11_04_2005/SofaCurrent/Projects/example1/Data/siege.sph", "r")
importSPH(fsph)
