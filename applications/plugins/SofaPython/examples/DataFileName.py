import Sofa
import sys

def createScene(node):

    # DataFileName
    objloader = node.createObject("MeshObjLoader",filename="mesh/snake_body.obj")
    print type(objloader.filename), objloader.filename, objloader.filename.fullPath


    # DataFileNameVector
    oglshader = node.createObject("OglShader")
    print type(oglshader.fileVertexShaders)


    sys.stdout.flush()