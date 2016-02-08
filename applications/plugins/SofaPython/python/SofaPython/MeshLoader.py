## In this file, we could add import from / export to
## mesh files in different formats

import os
import Sofa

class Mesh:

    def __init__(self):
        self.vertices = [] # vertices positions
        self.normals = [] # normals
        self.uv = [] # texture coordinates
        self.faceVertices = [] # indices in vertices
        self.faceNormals = [] # indices in normals
        self.faceUv = [] # indices in uv


def loadOBJ(filename):
    ## quickly written .obj import
    # Sofa.msg_info("PythonMeshLoader","loadOBJ: "+filename)

    m = Mesh()

    if not os.path.exists(filename):
        Sofa.msg_error("PythonMeshLoader","loadOBJ: inexistent file "+filename)
        return Mesh()

    for line in open(filename, "r"):
        vals = line.split()
        if len(vals)==0: # empty line
            continue
        if vals[0] == "v":
            v = map(float, vals[1:4])
            m.vertices.append(v)
        elif vals[0] == "vn":
            n = map(float, vals[1:4])
            m.normals.append(n)
        elif vals[0] == "vt":
            t = map(float, vals[1:3])
            m.uv.append(t)
        elif vals[0] == "f" or vals[0] == "l":
            faceV = []
            faceUV = []
            faceN = []
            for f in vals[1:]:
                w = f.split("/")
                faceV.append(int(w[0])-1)
                if len(w) > 1:
                    faceUV.append(int(w[1])-1)
                if len(w) > 2:
                    faceN.append(int(w[2])-1)
            m.faceVertices.append(faceV)
            m.faceUv.append(faceUV)
            m.faceNormals.append(faceN)

    return m