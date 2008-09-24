# -*- coding: iso8859-1 -*- 

# --------------------------------------------------------------------------
# 
# Exporte un objet Blender de type maillage dans un fichier obj
#  dans son repère local
#
# Ce programme est basé sur le script d'export off fourni avec Blender.
# Le script original des exports obj l'objet dans le repère global de la scène sans
#  tenir compte de l'orientation des maillages (rotations)
#
# Modification par Vincent Vansuyt
#
# Auteurs originaux :

#__author__ = "Anthony D'Agostino (Scorpius)"
#__url__ = ("blender", "blenderartists.org",
#"Author's homepage, http://www.redrival.com/scorpius")
#__version__ = "Part of IOSuite 0.5"

#
# +---------------------------------------------------------+
# | Copyright (c) 2002 Anthony D'Agostino                   |
# | http://www.redrival.com/scorpius                        |
# | scorpius@netzero.com                                    |
# | February 3, 2001                                        |
# | Read and write Object File Format (*.off)               |
# +---------------------------------------------------------+

# Et aussi :
#__author__ = "Campbell Barton, Jiri Hnidek"
#__url__ = ['www.blender.org', 'blenderartists.org']
#__version__ = "1.1"

# ***** BEGIN GPL LICENSE BLOCK *****
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.
#
# ***** END GPL LICENCE BLOCK *****

import Blender
import BPyMessages

# Python 2.3 has no reversed.
try:
	reversed
except:
	def reversed(l): return l[::-1]


class CExport_fichiers3D:
	MTL_DICT = {}
	
	# ==============================
	# ====== Write OBJ Format ======
	# ==============================
	def writeObj( self, object, filename ):
		if not object or object.type != 'Mesh':
			BPyMessages.Error_NoMeshActive()
			return
			
		print "Ecriture du fichier %s"%(filename)
		file = open(filename, 'wb')
		
		Blender.Window.WaitCursor(1)
		mesh = object.getData(mesh=1)

		# === OBJ Header ===
		# Write Header
		file.write('# Blender3D v%s OBJ File: %s\n' % (Blender.Get('version'),\
																	  Blender.Get('filename').split('/')[-1].split('\\')[-1] ))
		file.write('# www.blender3d.org\n')

		# === Vertex List ===
		for i, v in enumerate(mesh.verts):
			#file.write('%.6f %.6f %.6f\n' % tuple(v.co))
			file.write('v %.6f %.6f %.6f\n' % tuple(v.co))
		
		# === Transition entre les vertexs et les faces
		file.write('usemtl (null)\n') # mat, image
		file.write('s off\n')

		# === Face List ===
		for i, f in enumerate(mesh.faces):
			#file.write('%i' % len(f))
			file.write('f')
			for v in reversed(f.v):
				file.write(' %d' % (v.index + 1) )
			file.write('\n')
		
		file.close()
		Blender.Window.WaitCursor(0)
		message = 'Successfully exported "%s"' % Blender.sys.basename(filename)
		

#def fs_callback(filename):
#	write(filename)

#Blender.Window.FileSelector(fs_callback, "Export OFF", Blender.sys.makename(ext='.off'))
