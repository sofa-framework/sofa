#!BPY
"""Registration info for Blender menus:
Name: 'OBJ...'
Blender: 232
Group: 'Import'
Tip: 'Import Wavefront File Format (*.obj)'
"""

#=========================================================================
# Wavefront OBJ Importer/Exporter v1.2
# This is a complete OBJ importer and exporter script
# All Rights Reserved
# chris@artdreamdesigns.com 
#=========================================================================

#=========================================================================
# mise a jour pour Blender 228 et suivant                            jm soler
# mise a jour pour importer zbrush obj avec uvmapping                jm soler
# mise a jour pour utiliser le selecteur de fichier                  jm soler
# mise a jour pour importer les fichiers obj de forester/arboretum   jm soler
#=========================================================================

#=========================================================================
# DESCRIPTION: This script allows for full importing and exporting of
#      .obj files. uv texture coordinates and normals can be exported and
#      imported. .obj groups and materials can also be converted to and
#      from material indexes in Blender.
#
# INSTALLATION:
#      You need the FULL python installation to run this script. You can
#      down load the latest version of PYTHON from http://www.python.org.
#
# INSTRUCTIONS (You definitely want to read this!): 
#      Once the script is loaded in Blender, hit alt-p. This will bring up
#      the main interface panel. You will have a choise of exporting or
#      importing an .obj module. If you are exporting you must have at least
#      one mesh selected in Blender, or you will get an error.  
#      You can change the export filename by entering the path and filename
#      in the dialog.  If you do not enter a path, the path will default to
#      your blender directory. You can change the default path in the script <==== NOTE
#      itself by modifying the variable 'Filename' at the top of the script.  
#
#    EXPORTING:
#      There are 4 different export options: Default, Groups, Material Layers, 
#      and Standard. "Default" will export your mesh using Material Layers if
#      the mesh has material indexes. "Groups" and "Material Layers" are
#      logically equivalent, but are a different .obj format. If you are 
#      exporting a Poser morph target, you must select "Material Layers".   <===== NOTE
#      "Standard" will ignore grouping information, even if your mesh has 
#      material indexes.
#
#      There is also a choice between export using "mesh coordinates" or 
#      "object coordinates". "Object coordinates" are any location, rotation, 
#      or scaling values created outside of mesh edit. They belong to the object
#      rather than the mesh. If you export using mesh coordinates (the default)
#      the center of the object will always be at 0, 0, 0. Export using "mesh
#      coordinates is definintely what you want to use if you are working with
#      a Poser morph target. If you are exporting a group of objects, you will
#      automatically be placed in "object coordinate" mode. 
#
#    IMPORTING:
#      If your OBJ model has uv mapping coordinates, and you want to use them  <===== NOTE 
#      in Blender, you can access them in two ways. The best way is through Blender's
#      realtime UV coordinates which you enable simply by selecting the UV option in
#      the material edit window. This gives you an exact version of the uv coordinates.
#      An older method is to select the "stick" option in the material edit window. I
#      really don't know why anyone would want to use this option since it cannot handle
#      seams and texture overlap, but I left it in just in case someone could use it for
#      something.
#     
#      If your OBJ contains groups, once it has been imported, it may still appear
#      to lack any material indexes. If this happens, it can be remedied      <=== NOTE
#      by going to the mesh editor window, clicking on the mesh selection button, and
#      reselecting the mesh you have just imported. You will now have as many 
#      materials attached to your object as there are groups. You can then select 
#      different groups by doing a material select when you are in edit mode. 
#
#      Finally, you might have problems with certain parts of the object not displaying
#      after you go in and out of edit mode the first time. To fix this, simply go into
#      edit mode again, and select the "remove doubles" option.
#
#
# HISTORY:
#   Nov 13, 2001: Initial Release
#   Nov 16, 2001: Version 1.1 - no longer need to pre-define dummy materials
#   Dec 13, 2001: Version 1.2 - now imports into realtime UV (the UV button in the material edit window), and
#       exports realtime UV. This format is more compatible with the native .OBJ uv format. Should eliminate
#       texture misalignments and seams. Simply press the UV button in the material edit window after importing.
#
#  GetRaw
#================================


# ===============================
#   Setup our runtime constants
# ===============================

DEBUG=1         #Set this to "1" to see extra messages
MESHVERSION=3   # If the export file doesn't work,
FILEVERSION=3   # try changing these to "2"

EVENT_PATHCHANGE=     1
EVENT_IMPORT=         2
EVENT_IMPORT_CONT=    3
EVENT_OPTIONS=        4
EVENT_EXPORT=         7
EVENT_EXPORT_CHK=     5
EVENT_EXPORT_CANCEL=  6
EVENT_QUIT=           8
EVENT_EXPORT_ERR=     9
EVENT_TYPE=           10
EVENT_DONE=           11
EVENT_IMPORT_SELECT=  12

# ===============================
# Import our libraries
# ===============================

#import string
#import os
#import struct

try:
    import nt
    os=nt
    os.sep='\\'
except:    
    import posix
    os=posix
    os.sep='/'

def isdir(path):
    try:
        st = os.stat(path)
        return 1 
    except:
        return 0
    
def split(pathname):
         PATHNAME=pathname
         PATHNAME=PATHNAME.replace('\\','/') 
         k0=PATHNAME.split('/')
         directory=pathname.replace(k0[len(k0)-1],'')
         Name=k0[len(k0)-1]
         return directory, Name
        
def join(l0,l1):        
     return  l0+os.sep+l1
    
os.isdir=isdir
os.split=split
os.join=join

import math
import Blender
#import Blender210
from Blender import *
from Blender import NMesh
from Blender.Draw import *
from Blender.BGL import *
from Blender import Material
from Blender import Window



# ===============================
# Input Variables
# ===============================

Filename = "G:\\tmp\\test.obj"

gFilename=Create(Filename)
gAlert   = 0
type     = 1
exporttype = 1
returncode = 0
operation = "Export"
center = [0,0,0]
rotation = [0,0,0]
Transform = []
multiflag = 0

#================================
# def Fileselect function:
#================================
def ImportFunctionselet(filename):
             global gFilename 
             global ExportOptions
             global ExportType
             global type
             global exporttype
             global operation
             global gAlert
             gFilename.val=filename
             ImportFunction(filename, type)
             operation = "Import"

#================================
def ExitGUI ():
#================================
    Exit()

#================================
def EventGUI (event):
#================================
      global gFilename 
      global ExportOptions
      global ExportType
      global type
      global exporttype
      global operation
      global gAlert

      if (event==EVENT_IMPORT):
         ImportFunction(gFilename.val, type)
         operation = "Import"

      if (event==EVENT_IMPORT_SELECT):         
         Window.FileSelector (ImportFunctionselet, 'IMPORT FILE')


      if (event==EVENT_IMPORT_CONT):
         gAlert = 0
         operation = "Import"
         Draw ()

      if (event==EVENT_EXPORT):
         ExportFunction(gFilename.val, type)
         operation = "Export"

      if (event==EVENT_EXPORT_CHK):
         ExportFunctionOK(gFilename.val, type)
         Draw ()
      if (event==EVENT_EXPORT_CANCEL):
         gAlert = 0
         Draw ()
      if (event==EVENT_OPTIONS):
         type = ExportOptions.val
         Draw ()
      if (event==EVENT_TYPE):
         exporttype = ExportType.val
         Draw ()
      if (event==EVENT_EXPORT_ERR):
         gAlert = 0
         Draw ()
      if (event==EVENT_DONE):  
         gAlert = 0
         Draw ()
      if (event==EVENT_QUIT):
         ExitGUI()

#================================
def DrawGUI():
#================================
      global type
      global exporttype
      global operation 

      glClearColor (0.6,0.6,0.6,0)
      glClear (GL_COLOR_BUFFER_BIT)

      global gFilename
      global gAlert
      global ExportOptions
      global ExportType

      if (gAlert==0):
         # Add in the copyright notice and title
         glRasterPos2d(32, 380)
         Text("Wavefront OBJ Importer/Exporter")
         glRasterPos2d(32, 350)
         Text("Copyright (C) Chris Lynch 2001")

         gFilename=String ("Filename: ",EVENT_PATHCHANGE,32,250,320,32,gFilename.val,255,"Full pathname and filename")
         Button ("Export",EVENT_EXPORT,32,200,100,32)
         Button ("Import",EVENT_IMPORT,252,200,100,32)
         Button ("Select Import",EVENT_IMPORT_SELECT,355,200,100,32)
         glRasterPos2d(32, 165)
         Text("Select Export Options:")
         options = "Export Options %t| Default %x1| Material Layers %x2| Obj Groups %x3| Standard %x4"
         ExportOptions = Menu (options,EVENT_OPTIONS,200,150,150,32, type)
         Button ("Done",EVENT_QUIT,142,50,100,32)
         glRasterPos2d(32, 115)
         Text("Export using ")
         options = "Export Type %t| Mesh Coordinates %x1| Object Coordinates %x2"
         ExportType = Menu (options,EVENT_TYPE,170,100,180,32, exporttype)
         Button ("Done",EVENT_QUIT,142,50,100,32)

      elif (gAlert==1):
         glRasterPos2i (32,250)
         Text (gFilename.val+ " already exists. Save anyway?")
         Button ("Save",EVENT_EXPORT_CHK,150,200,50,32)
         Button ("Cancel",EVENT_EXPORT_CANCEL,250,200,50,32)
         gAlert = 0
      elif (gAlert==2):
         glRasterPos2i (32,250)
         Text (gFilename.val+ " cannot be found. Check directory and filename.")
         Button ("Continue",EVENT_IMPORT_CONT,32,190,70,32) 
         gAlert = 0
      elif gAlert == 3:
         glRasterPos2i (32,250)
         Text ("No objects selected to export. You must select one or more objects.")
         Button ("Continue",EVENT_EXPORT_ERR,192,200,70,32)
         gAlert = 0
      elif gAlert == 5:
         glRasterPos2i (32,250)
         Text ("Invalid directory path.")
         Button ("Continue",EVENT_EXPORT_ERR,192,200,70,32)
         gAlert = 0
      else:
         glRasterPos2i (32,250)
         Text (str(operation)+ " of " +str(gFilename.val)+ " done.")
         Button ("Continue",EVENT_DONE,192,200,70,32)
         
#================================
def RegisterGUI ():
#================================
    Register (DrawGUI,None,EventGUI)

#================================
# MAIN SCRIPT
#================================
# Opens a file, writes data in it
# and closes it up.
#================================
RegisterGUI()

#================================
def ImportFunction (importName, type):
#================================       
      global gFilename 
      global gAlert

      try:
         FILE=open (importName,"r")
         directory, Name = os.split(gFilename.val)
         print directory, Name
         words = Name.split(".")
         Name = words[0]
         ObjImport(FILE, Name, gFilename.val) 
         FILE.close()
         gAlert = 4
         Draw ()
      except IOError:
         gAlert=2
         Draw ()

#================================
def ExportFunction (exportName, type):
#================================       
      global gFilename 
      global gAlert

      try:
         FILE=open (exportName,"r")
         FILE.close()
         gAlert = 1
         Draw ()
      except IOError:

         directory, Name = os.split(gFilename.val)

         
         if os.isdir(directory):
            ExportFunctionOK(exportName, type)
            Draw ()
         else:
            gAlert = 5
            Draw ()
 
#================================
def ExportFunctionOK (exportName, type):
#================================       
      global gFilename 
      global gAlert
      global returncode

      FILE=open (exportName,"w")

      directory, Name = os.split(gFilename.val)
      
      words = Name.split(".")
      Name = words[0]
      ObjExport(FILE, Name, type)
      if returncode > 0:
         gAlert = 3
      else:
         gAlert = 4
      FILE.flush()
      FILE.close()

#=========================
def ObjImport(file, Name, filename):
#=========================   
    vcount     = 0
    vncount    = 0
    vtcount    = 0
    fcount     = 0
    gcount     = 0
    setcount   = 0
    groupflag  = 0
    objectflag = 0
    mtlflag    = 0
    baseindex  = 0
    basevtcount = 0
    basevncount = 0
    matindex   = 0

    pointList    = []
    uvList       = []
    normalList   = []
    faceList     = []
    materialList = []
    imagelist    = []
    
    uv = [] 
    lines = file.readlines()
    linenumber = 1

    for line in lines:
        words = line.split()
        if words and words[0] == "#":
            pass # ignore comments
        elif words and words[0] == "v":
            vcount = vcount + 1
            
            for n_ in [1,2,3]: 
               if words[n_].find(',')!=-1:
                    words[n_]=words[n_].replace(',','.')
                
            x = float(words[1])
            y = float(words[2])
            z = float(words[3])

            pointList.append([x, y, z])

        elif words and words[0] == "vt":
            vtcount = vtcount + 1
            for n_ in [1,2]: 
               if words[n_].find(',')!=-1:
                    words[n_]=words[n_].replace(',','.')

            u = float(words[1])
            v = float(words[2])
            uvList.append([u, v])

        elif words and words[0] == "vn":
            vncount = vncount + 1

            for n_ in [1,2,3]: 
               if words[n_].find(',')!=-1:
                    words[n_]=words[n_].replace(',','.')

            i = float(words[1])
            j = float(words[2])
            k = float(words[3])
            normalList.append([i, j, k])

        elif words and words[0] == "f":
            fcount = fcount + 1
            vi = [] # vertex  indices
            ti = [] # texture indices
            ni = [] # normal  indices
            words = words[1:]
            lcount = len(words)
            for index in (xrange(lcount)):
               if words[index].find( "/") == -1:
                     vindex = int(words[index])
                     if vindex < 0: vindex = baseindex + vindex + 1  
                     vi.append(vindex)
               else:
                   vtn = words[index].split( "/")
                   vindex = int(vtn[0])
                   if vindex < 0: vindex = baseindex + vindex + 1 
                   vi.append(vindex) 
            
                   if len(vtn) > 1 and vtn[1]:
                      tindex = int(vtn[1])
                      if tindex < 0: tindex = basevtcount +tindex + 1
                      ti.append(tindex)

                   if len(vtn) > 2 and vtn[2]:
                      nindex = int(vtn[2])
                      if nindex < 0: nindex = basevncount +nindex + 1
                      ni.append(nindex)
            faceList.append([vi, ti, ni, matindex])

        elif words and words[0] == "o":
            ObjectName = words[1]
            objectflag = 1
            #print "Name is %s" % ObjectName

        elif words and words[0] == "g":
            groupflag = 1
            index = len(words)
            if objectflag == 0:
               objectflag = 1
               if index > 1:
                  ObjectName = words[1].join("_")
                  GroupName = words[1].join("_") 
               else:
                  ObjectName = "Default" 
                  GroupName = "Default" 
               #print "Object name is %s" % ObjectName
               #print "Group name is %s" % GroupName
            else:
               if index > 1:
                  GroupName = join(words[1],"_") 
               else:
                  GroupName = "Default" 
               #print "Group name is %s" % GroupName
                  
            if mtlflag == 0:
               matindex = AddMeshMaterial(GroupName,materialList, matindex)
            gcount = gcount + 1 
               
            if fcount > 0: 
               baseindex = vcount
               basevncount = vncount
               basevtcount = vtcount

        elif words and words[0] == "mtllib":
            # try to export materials
            directory, dummy = os.split(filename)
            filename = os.join(directory, words[1])
            print  "try to import : ",filename
            
            try:
                file = open(filename, "r")
            except:
                print "no material file %s" % filename
            else:
                mtlflag = 0
                #file = open(filename, "r")
                line = file.readline()
                mtlflag = 1
                while line:
                    words = line.split()
                    if words and words[0] == "newmtl":
                      name = words[1]
                      line = file.readline()  # Ns ?
                      words = line.split()
                      while words[0] not in ["Ka","Kd","Ks","map_Kd"]:
                          line = file.readline()
                          words = line.split()
                             
                      if words[0] == "Ka":
                        Ka = [float(words[1]),
                              float(words[2]),
                              float(words[3])]
                        line = file.readline()  # Kd
                        words = line.split()
                        
                      if words[0] == "Kd":
                        Kd = [float(words[1]),
                              float(words[2]),
                              float(words[3])]
                        line = file.readline()  # Ks 
                        words = line.split()
                         
                      if words[0] == "Ks":
                        Ks = [float(words[1]),
                              float(words[2]),
                              float(words[3])]
                        line = file.readline()  # Ks 
                        words = line.split()
                        
                      if words[0] == "map_Kd":
                        Kmap= words[1]
                        img=os.join(directory, Kmap)
                        im=Blender.Image.Load(img)
                        line = file.readline()  # Ks 
                        words = line.split()
                          
                      matindex = AddGlobalMaterial(name, matindex)                            
                      matlist = Material.Get() 
                        
                      if len(matlist) > 0:
                         if name!='defaultMat':
                             material = matlist[matindex]
                             material.R = Kd[0]
                             material.G = Kd[1]
                             material.B = Kd[2]
                             try:
                                  material.specCol[0] = Ks[0]
                                  material.specCol[1] = Ks[1]
                                  material.specCol[2] = Ks[2]
                             except:
                                  pass
                             try:
                                  alpha = 1 - ((Ka[0]+Ka[1]+Ka[2])/3)
                             except:
                                  pass
                             try:
                                  material.alpha = alpha
                             except:
                                  pass

                             try:
                                 
                                 img=os.join(directory, Kmap)
                                 im=Blender.Image.Load(img)
                                 imagelist.append(im)
                             
                                 t=Blender.Texture.New(Kmap)
                                 t.setType('Image')
                                 t.setImage(im)
                             
                                 material.setTexture(0,t)
                                 material.getTextures()[0].texco=16
                             except:
                                  pass
                                
                         else:
                             material = matlist[matindex]
                             
                             material.R = 0.8
                             material.G = 0.8
                             material.B = 0.8
                             material.specCol[0] = 0.5
                             material.specCol[1] = 0.5
                             material.specCol[2] = 0.5
                             
                             img=os.join(directory, Kmap)
                             im=Blender.Image.Load(img)
                             imagelist.append(im)
                             
                             t=Blender.Texture.New(Kmap)
                             t.setType('Image')
                             t.setImage(im)
                             
                             material.setTexture(0,t)
                             material.getTextures()[0].texco=16
                       
                      else:
                         mtlflag = 0
                             
                    line = file.readline()
                          
                        
                file.close()
                 
        elif words and words[0] == "usemtl":
            if mtlflag == 1:
               name = words[1]
               matindex = AddMeshMaterial(name, materialList, matindex) 
        elif words:   
            print "%s: %s" % (linenumber, words)
        linenumber = linenumber + 1
    file.close()

    # import in Blender
 
    print "import into Blender ..."
    mesh   = NMesh.GetRaw ()

    i = 0
    while i < vcount:
      x, y, z = pointList[i] 
      vert=NMesh.Vert(x, y, z)
      mesh.verts.append(vert)
      i=i+1

    if vtcount > 0:
       #mesh.hasFaceUV() = 1
       print ("Object has uv coordinates")
 
    if len(materialList) > 0:
       for m in materialList:
          try:
            M=Material.Get(m)
            mesh.materials.append(M) 
          except:
            pass

    total = len(faceList)
    i = 0

    for f in faceList:
        if i%1000 == 0:
          print ("Progress = "+ str(i)+"/"+ str(total))

        i = i + 1
        vi, ti, ni, matindex = f
        face=NMesh.Face()
        if len(materialList) > 0:
           face.mat = matindex

        limit = len(vi)
        setcount = setcount + len(vi)
        c = 0    
    
        while c < limit:
          m = vi[c]-1
          if vtcount > 0 and len(ti) > c:
             n = ti[c]-1
          if vncount > 0 and len(ni) > c:
             p = ni[c]-1

          if vtcount > 0:
             try:
                  u, v = uvList[n]
             except:
                  pass 

             """ 
        #  multiply uv coordinates by 2 and add 1. Apparently blender uses uv range of 1 to 3 (not 0 to 1). 
             mesh.verts[m].uvco[0] = (u*2)+1
             mesh.verts[m].uvco[1] = (v*2)+1
            """

          if vncount > 0:
             if p > len(normalList):
                print("normal len = " +str(len(normalList))+ " vector len = " +str(len(pointList)))
                print("p = " +str(p))
             x, y, z = normalList[p]  
             mesh.verts[m].no[0] = x
             mesh.verts[m].no[1] = y
             mesh.verts[m].no[2] = z
          c = c+1  
      
        if len(vi) < 5:
          for index in vi:
            face.v.append (mesh.verts[index-1])
  
          if vtcount > 0:  
            for index in ti:
               u, v = uvList[index-1]
               face.uv.append((u,v))
               
            if len(imagelist)>0:
                face.image=imagelist[0]
                #print
                
          if vcount>0:
             face.smooth=1

          mesh.faces.append(face) 

    print "all other (general) polygons ..."
    for f in faceList:
        vi, ti, ni, matindex = f 
        if len(vi) > 4:
            # export the polygon as edges
            print ("Odd face, vertices = "+ str(len(vi)))
            for i in range(len(vi)-2):
               face = NMesh.Face()
               if len(materialList) > 0:
                  face.mat = matindex
               face.v.append(mesh.verts[vi[0]-1])
               face.v.append(mesh.verts[vi[i+1]-1])
               face.v.append(mesh.verts[vi[i+2]-1])

               if vtcount > 0: 
                  if len(ti) > i+2:
                     u, v = uvList[ti[0]-1]
                     face.uv.append((u,v))
                     u, v = uvList[ti[i+1]-1]
                     face.uv.append((u,v))
                     u, v = uvList[ti[i+2]-1]
                     face.uv.append((u,v))

               mesh.faces.append(face)
      
    NMesh.PutRaw(mesh, Name,1)

    print ("Total number of vertices is "+ str(vcount))
    print ("Total number of faces is "+ str(len(faceList)))
    print ("Total number of sets is "+ str(setcount))


    print("Finished importing " +str(Name)+ ".obj")

#=========================================
def AddMeshMaterial(name, materialList, matindex):
#=========================================
    
   index = 0
   found = 0 
   limit = len(materialList)

   while index < limit:
     if materialList[index] == name:
        matindex = index 
        found = 1
        index = limit
     index = index + 1
   
   if found == 0:      
      materialList.append(name)
      matindex = len(materialList)-1
        
   return matindex

#=========================================
def AddGlobalMaterial (name, matindex):
#=========================================
    
   index = 0
   found = 0
   matindex  = 0
   MatList = Material.Get()
   limit = len(MatList)

   while index < limit:
     if MatList[index].name == name:
        matindex = index 
        found = 1
        index = limit
     index = index + 1

   if found == 0:
      material = Material.New(name)
      matindex = index
    
   return matindex

#================================
def ObjExport(FILE, Name, type):
#================================
  global returncode
  global vertexcount
  global uvcount
  global Transform
  global multiflag
  global exporttype

  vertexcount = 0
  uvcount = 0
  returncode = 0
  print("Writing %s..." % Name)
  FILE.write("# Wavefront OBJ (1.0) exported by lynx's OBJ import/export script\n\n")

  Objects = Object.GetSelected()
  if Objects == []:
     print("You have not selected an object!")
     returncode = 4
  else:
     for object in Objects:
        MtlList = []
        if len(Objects) > 1 or exporttype > 1:
           Transform = CreateMatrix(object, Transform)
           multiflag = 1
           
        mesh = NMesh.GetRawFromObject(object.name)
        ObjName = mesh.name
        has_uvco = mesh.hasVertexUV()

        FILE.write("# Meshname:\t%s\n" % ObjName)

        faces = mesh.faces
        materials = mesh.materials
        Vertices = mesh.verts
        GlobalMaterials = Material.Get()

        if len(materials) >= 1 and len(GlobalMaterials) > 0 and type < 4:
           CreateMtlFile(Name, materials, MtlList)

        # Total Vertices and faces; comment if not useful
        FILE.write("# Total number of Faces:\t%s\n" % len(faces))
        FILE.write("# Total number of Vertices:\t%s\n" % len(Vertices))

        FILE.write("\n")

        # print first image map for uvcoords to use
        # to be updated when we get access to other textures
        if mesh.hasFaceUV(): FILE.write("# UV Texture:\t%s\n\n" % mesh.hasFaceUV())

        if len(materials) >= 1 and len(GlobalMaterials) > 0 and type < 3:
           UseLayers(faces, Vertices, MtlList, has_uvco, FILE, ObjName, Name)
        elif len(materials) >= 1 and len(GlobalMaterials) > 0 and type == 3:
           UseMtl(faces, Vertices, MtlList, has_uvco, FILE, ObjName, Name)
        else:
           Standard(faces, Vertices, has_uvco, FILE, ObjName)
 
#================================================
def CreateMtlFile (name, MeshMaterials, MtlList):
#================================================
      global gFilename 

    # try to export materials
      directory, mtlname = os.split(gFilename.val)
      mtlname = name + ".mtl"
      filename = os.join(directory, mtlname)
      file = open(filename, "w")

      file.write("# Materials for %s.\n" % (name + ".obj"))
      file.write("# Created by Blender.\n")
      file.write("# These files must be in the same directory for the materials to be read correctly.\n\n")

      MatList = Material.Get()
      print str(MeshMaterials)

      MtlNList=[]
      for m in  MatList:
         MtlNList.append(m.name)

      counter = 1
      found = 0  

      for material in MeshMaterials:
         for mtl in MtlList:
            if material == mtl:
                found = 1

         MtlList.append(material) 

         if found == 0:
            file.write("newmtl %s \n" % material.name)
            index = 0
            print material, MatList
            while index < len(MatList):
               if material.name == MatList[index].name:
                  mtl = MatList[index]
                  index = len(MatList)
                  found = 1
               index = index + 1

            if found == 1:
               alpha = mtl.getAlpha()
               file.write("       Ka %s %s %s \n" % (round(1-alpha,5), round(1-alpha,5), round(1-alpha,5)))
               file.write("       Kd %s %s %s \n" % (round(mtl.R,5), round(mtl.G,5), round(mtl.B,5)))
               file.write("       Ks %s %s %s \n" % (round(mtl.specCol[0],5), round(mtl.specCol[1],5), round(mtl.specCol[2],5)))
               mtextures = mtl.getTextures()           # get a list of the MTex objects
               try:
                 for mtex in mtextures:
                     if mtex.tex.type == Texture.Types.IMAGE and (mtex.texco & Texture.TexCo.UV):
                             file.write("       map_Kd %s \n" % Blender.sys.basename(mtex[0].tex.image.filename))
                             break
               except:
                 if mtextures[0].tex.type == Texture.Types.IMAGE and (mtextures[0].texco & Texture.TexCo.UV):
                             file.write("       map_Kd %s \n" % Blender.sys.basename(mtextures[0].tex.image.filename))
                           
                   
               file.write("       illum 1\n")
               
            else:
               file.write("       Ka %s %s %s \n" % (0, 0, 0))
               file.write("       Kd %s %s %s \n" % (1, 1, 1))
               file.write("       Ks %s %s %s \n" % (1, 1, 1))
               file.write("       illum 1\n")

         found = 0

      file.flush()
      file.close()
 
#===========================================================
def Standard(faces, Vertices, has_uvco, FILE, ObjName): 
#=========================================================== 
       global vertexcount
       global uvcount
       global multiflag

       uvPtrs = []
       uvList = []

       FILE.write("o %s\n\n" % (ObjName)) 
       FILE.write("g %s\n\n" % (ObjName)) 
 
       for v in Vertices: 
          vert = v.co 
          if multiflag  == 1:
             vert = Alter(vert, Transform) 
          x, y, z = vert
               
          FILE.write("v %s %s %s\n" % (x, y, z))

       uv_flag = 0
       for face in faces:
         for uv in face.uv:
            found = 0
            index = len(uvList)
            limit = 0
            if len(uvList)-200 > 0:
               limit = len(uvList)-200
            while index > limit and found == 0:
               uv_value = uvList[index-1]
               if uv[0] == uv_value[0] and uv[1] == uv_value[1]:
                  uvPtrs.append(index+uvcount)
                  found = 1
               index = index - 1
            if found == 0:
               uvList.append(uv)
               index = len(uvList)
               uvPtrs.append(index+uvcount)
               u, v = uv
               FILE.write("vt %s %s\n" % (u, v))
               uv_flag = 1

       if has_uvco and uv_flag == 0:
         for v in Vertices:
            u, v, z = v.uvco 
            u = (u-1)/2
            v = (v-1)/2
            FILE.write("vt %s %s\n" % (u, v))

       for v in Vertices: 
          x, y, z = v.no
          FILE.write("vn %s %s %s\n" % (x, y, z))

       p = 0
       uvindex = 0
       total = len(faces)

       for face in faces:
          p = p+1
          if (p%1000) == 0:
              print ("Progress = "+ str(p)+ " of "+ str(total) +" faces")

          FILE.write("f ")
          for index in range(len(face.v)):
             v = face.v[index].index + vertexcount
             if len(face.uv) > 0:
                FILE.write("%s/%s/%s " % (v+1, uvPtrs[uvindex], v+1))
                uvindex = uvindex+1
             elif has_uvco:
                FILE.write("%s/%s/%s " % (v+1, v+1, v+1))
             else:                     
                FILE.write("%s//%s " % (v+1, v+1))
          FILE.write("\n")

       vertexcount = vertexcount + len(Vertices)
       uvcount = uvcount + len(uvList)

       print("Export of " +str(ObjName)+ ".obj finished.\n")

#=====================================================================
def UseLayers(faces, Vertices, MtlList, has_uvco, FILE, ObjName, Name): 
#===================================================================== 
       global vertexcount
       global uvcount
       global multiflag

       uvPtrs = []
       uvList = []

       FILE.write("mtllib %s\n\n" % (Name + ".mtl"))
       FILE.write("g %s\n\n" % (ObjName)) 

       for v in Vertices: 
          vert = v.co 
          if multiflag  == 1:
             vert = Alter(vert, Transform)   
          x, y, z = vert
          FILE.write("v %s %s %s\n" % (x, y, z))

       uv_flag = 0
       for m in range(len(MtlList)):
          for face in faces:
              if face.mat == m:
                 for uv in face.uv:
                    found = 0
                    index = len(uvList)
                    limit = 0
                    if len(uvList)-200 > 0:
                       limit = len(uvList)-200
                    while index > limit and found == 0:
                       uv_value = uvList[index-1]
                       if uv[0] == uv_value[0] and uv[1] == uv_value[1]:
                          uvPtrs.append(index+uvcount)
                          found = 1
                       index = index - 1
                    if found == 0:
                       uvList.append(uv)
                       index = len(uvList)
                       uvPtrs.append(index+uvcount)
                       u, v = uv
                       FILE.write("vt %s %s\n" % (u, v))
                       uv_flag = 1

       if has_uvco and uv_flag == 0:
         for v in Vertices:
            u, v, z = v.uvco
            u = (u-1)/2
            v = (v-1)/2
            FILE.write("vt %s %s\n" % (u, v))

       for v in Vertices: 
          x, y, z = v.no
          FILE.write("vn %s %s %s\n" % (x, y, z))

       total = len(faces)
       p = 0
       uvindex = 0
       for m in range(len(MtlList)):         
          FILE.write("usemtl %s\n" % (MtlList[m].name)) 
          for face in faces:
              if face.mat == m:
                p = p+1
                if (p%1000) == 0:
                   print ("Progress = "+ str(p)+ " of "+ str(total) +" faces")

                FILE.write("f ")
                for index in range(len(face.v)):
                   v = face.v[index].index + vertexcount 
                   if len(face.uv) > 0:
                      FILE.write("%s/%s/%s " % (v+1, uvPtrs[uvindex], v+1))
                      uvindex = uvindex+1
                   elif has_uvco:
                      FILE.write("%s/%s/%s " % (v+1, v+1, v+1))
                   else:
                      FILE.write("%s//%s " % (v+1, v+1))
                FILE.write("\n")

       vertexcount = vertexcount + len(Vertices) 
       print("Export of " +str(ObjName)+ ".obj using material layers finished.\n")

#==================================================================
def UseMtl(faces, Vertices, MtlList, has_uvco, FILE, ObjName, Name): 
#==================================================================
       global vertexcount
       global multiflag

       FILE.write("mtllib %s\n\n" % (Name + ".mtl")) 
       FILE.write("o %s\n\n" % (ObjName))
       
       index = 0
       VertexList = []
       for vertex in Vertices:
          VertexList.append(-1)
          index = index + 1
       print("number of vertices is " +str(len(VertexList)))

       Totalindex = 0
       ix = 0
       NewVertexList = []
       NewVertexCo = []
       for m in range(len(MtlList)):
           # Group name is the name of the mesh 
           if MtlList[m]:
              FILE.write("g %s\n" % (MtlList[m].name+str(m+1))) 
           else:
              FILE.write("g %s\n" % ("Null"+str(m+1)))
           FILE.write("s off\n\n") 
         
           FILE.write("usemtl %s\n\n" % (MtlList[m].name)) 
 
           for face in faces:
              if face.mat == m:
                 for vertex in face.v:
                    v = vertex.index 
                    if VertexList[v] < 0:
                       VertexList[v] = Totalindex
                       NewVertexList.append(v)
                       Totalindex = Totalindex + 1
 
           for v_old in NewVertexList:
              vert = Vertices[v_old].co
              if multiflag  == 1:
                vert = Alter(vert, Transform)
              x, y, z = vert
              FILE.write("v %s %s %s\n" % (x, y, z))
              NewVertexCo.append([x,y,z])

           if has_uvco:
              for v_old in NewVertexList:
                 u, v, z = Vertices[v_old].uvco
                 u = (u-1)/2
                 v = (v-1)/2              
                 FILE.write("vt %s %s\n" % (u, v))

           for v_old in NewVertexList:
              x, y, z = Vertices[v_old].no
              FILE.write("vn %s %s %s\n" % (x, y, z))
  
           for face in faces:
             if face.mat == m:
                FILE.write("f ")
                for index in range(len(face.v)):
                   v = face.v[index].index
                   v_new = VertexList[v] 
                   if has_uvco:
                      FILE.write("%s/%s/%s " % (v_new+1, v_new+1, v_new+1))
                   else:
                      FILE.write("%s//%s " % (v_new+1, v_new+1))
                FILE.write("\n")

           FILE.write("\n")

           NewVertexList = []
           print("Group " +str(m+1)+ " of " +str(len(MtlList))+ " finished.")
 
       print("Export of " +str(ObjName)+ ".obj using groups finished.\n")

#========================================
def CreateMatrix(object, Transform):
#========================================
   Mx = []
   My = []
   Mz = []
   T1 = []
   Transform = []

   angle = object.RotX
   Mx.append([1, 0, 0])
   y = math.cos(angle)
   z = -math.sin(angle)
   Mx.append([0, y, z])
   y = math.sin(angle)
   z = math.cos(angle)
   Mx.append([0, y, z])

   angle = object.RotY
   x = math.cos(angle)
   z = math.sin(angle)
   My.append([x, 0, z])
   My.append([0, 1, 0])
   x = -math.sin(angle)
   z = math.cos(angle)
   My.append([x, 0, z])

   angle = object.RotZ
   x = math.cos(angle)
   y = -math.sin(angle)
   Mz.append([x, y, 0])
   x = math.sin(angle)
   y = math.cos(angle)
   Mz.append([x, y, 0])
   Mz.append([0, 0, 1])

   m0 = Mx[0]
   m1 = Mx[1]
   m2 = Mx[2]
   for row in My:
      x, y, z = row
      nx = x*m0[0] + y*m1[0] + z*m2[0]
      ny = x*m0[1] + y*m1[1] + z*m2[1]
      nz = x*m0[2] + y*m1[2] + z*m2[2]
      T1.append([nx, ny, nz])

   m0 = T1[0]
   m1 = T1[1]
   m2 = T1[2]
   for row in Mz:
     x, y, z = row
     nx = x*m0[0] + y*m1[0] + z*m2[0]
     ny = x*m0[1] + y*m1[1] + z*m2[1]
     nz = x*m0[2] + y*m1[2] + z*m2[2]
     Transform.append([nx, ny, nz])

   Transform.append([object.SizeX, object.SizeY, object.SizeZ])
   Transform.append([object.LocX, object.LocY, object.LocZ])

   return Transform

#======================================
def Alter(vect, Transform):
#======================================
   v2 = []
   nv = []

   x, y, z = vect
   sx, sy, sz = Transform[3]
   lx, ly, lz = Transform[4]

   v2.append(x*sx)
   v2.append(y*sy)
   v2.append(z*sz)

   for index in range(len(vect)):
      t = Transform[index]
      nv.append(v2[0]*t[0] + v2[1]*t[1] +v2[2]*t[2])

   nv[0] = nv[0]+lx
   nv[1] = nv[1]+ly
   nv[2] = nv[2]+lz

   return nv
