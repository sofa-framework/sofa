import os
import math
import sys

def loadOBJ(filename):
    # Check if file exist
    if filename == "" or not filename: return

    filename = os.path.realpath(filename)
    if not os.path.exists(filename):
        return

    # Output data
    vertices = []
    normals = []
    texcoords = []
    faces = []

    # Init the append for everything
    vertices_append = vertices.append
    normals_append = normals.append
    texcoords_append = texcoords.append
    faces_append = faces.append

    # Begin loop
    file_content = open(filename, "r")
    for line in file_content:
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            vertices_append(map(float, values[1:4]))
        elif values[0] == 'vn':
            normals_append(map(float, values[1:4]))
        elif values[0] == 'vt':
            texcoords_append(map(float, values[1:3]))
        elif values[0] == 'f':
            face = list(); tcoords = list(); norms = list()
            f_append = face.append; t_append = tcoords.append; n_append = norms.append
            for v in values[1:]:
                w = v.split('/')
                f_append(int(w[0])-1)
                if len(w) > 1 and len(w[1]) > 0:
                    t_append(int(w[1])-1)
                else:
                    t_append(0)
                if len(w) > 2 and len(w[2]) > 0:
                    n_append(int(w[2])-1)
                else:
                    n_append(0)
            faces_append((face, norms, tcoords))
    return (vertices, normals, texcoords, faces)
